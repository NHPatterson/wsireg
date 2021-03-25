from pathlib import Path
import SimpleITK as sitk
import itk
from wsireg.utils.im_utils import (
    read_image,
    std_prepro,
    contrast_enhance,
    sitk_inv_int,
    get_im_info,
    guess_rgb,
    transform_to_ome_tiff,
    transform_to_ome_zarr,
    prepare_np_image,
)
from wsireg.utils.tform_utils import (
    transform_2d_image_itkelx,
    gen_aff_tform_flip,
    gen_rigid_tform_rot,
    wsireg_transforms_to_itk_composite,
)
from wsireg.utils.reg_utils import json_to_pmap_dict
from wsireg.utils.itk_im_conversions import (
    sitk_image_to_itk_image,
)


class RegImage:
    """
    Image container class for reading images and preparing them for registration.
    Does two forms of preprocessing:
    Intensity preprocessing, modifying intensity values for best registration
    Spatial preprocessing, rotating and flipping images prior to registration for best alignment

    After preprocessing, images destined for registration should always be a single channel 2D image

    Parameters
    ----------
    image : str or np.array
        filepath to image to be processed or a numpy array of the image
    image_res : float
        Physical spacing (xy resolution) of the image (units are assumed to be the same for each image in experiment but are not
        defined)
        Does not yet support isotropic resolution
    prepro_dict : dict
        preprocessing dict to apply to image
    transforms:list
        list of elastix transformation data to apply to an image during preprocessing
    mask : str or np.ndarray
        filepath of a mask image to be processed or a numpy array of the mask
        all values => 1 are considered WITHIN the mask
    """

    def __init__(
        self,
        image,
        image_res,
        prepro_dict={
            "image_type": "FL",
            "ch_indices": None,
            "as_uint8": False,
        },
        transforms=None,
        mask=None,
    ):
        if isinstance(image, str):
            self.image = Path(image)
        else:
            self.image = image

        self.image_res = image_res
        self.preprocessing = std_prepro()
        self.transforms = [] if transforms is None else transforms

        if mask is not None:
            if isinstance(mask, str):
                self.mask = read_image(mask, preprocessing=None)
            else:
                self.mask = prepare_np_image(mask, preprocessing=None)
            self.mask.SetSpacing((self.image_res, self.image_res))
        else:
            self.mask = None

        if prepro_dict is None:
            if isinstance(self.image, Path):
                self.image = read_image(self.image, preprocessing=None)
            else:
                self.image = prepare_np_image(self.image, preprocessing=None)

            if self.image.GetDepth() > 0:
                self.image.SetSpacing((self.image_res, self.image_res, 1))
            else:
                self.image.SetSpacing((self.image_res, self.image_res))

        else:

            for k, v in prepro_dict.items():
                self.preprocessing[k] = v

            (
                self.image,
                spatial_preprocessing,
            ) = self.preprocess_reg_image_intensity(
                self.image, self.preprocessing
            )

            if self.image.GetDepth() >= 1:
                raise ValueError(
                    "preprocessing did not result in a single image plane"
                )

            if len(spatial_preprocessing) > 0 or transforms is not None:
                self.preprocess_reg_image_spatial(
                    spatial_preprocessing, transforms
                )

        if self.image.GetDepth() > 0:
            self.image.SetOrigin((0, 0, 0))
        else:
            self.image.SetOrigin((0, 0))
        if self.mask is not None:
            self.mask.SetOrigin((0, 0))

        self.pixel_id = self.image.GetPixelID()

    def preprocess_reg_image_intensity(self, image, preprocessing):
        if isinstance(image, Path):
            image = read_image(str(image), preprocessing=preprocessing)
        else:
            image = prepare_np_image(image, preprocessing=preprocessing)

        # separate spatial preprocessing from intensity preprocessing
        spatial_preprocessing = {}
        for spat_key in ["mask_bbox", "rot_cc", "flip"]:
            if spat_key in preprocessing:
                spatial_preprocessing[spat_key] = preprocessing[spat_key]
                preprocessing.pop(spat_key, None)

        # remove read time preprocessing
        preprocessing.pop("ch_indices", None)
        preprocessing.pop("as_uint8", None)

        # type specific
        if preprocessing["image_type"] == "FL":
            preprocessing.pop("inv_int", None)
        elif preprocessing["image_type"] == "BF":
            preprocessing.pop("max_int_proj", None)
            preprocessing.pop("contrast_enhance", None)

        if preprocessing.get("contrast_enhance_opt") is True:
            preprocessing.update({"contrast_enhance": contrast_enhance})
        else:
            preprocessing.pop("contrast_enhance", None)

        if preprocessing.get("inv_int_opt") is True:
            preprocessing.update({"inv_int": sitk_inv_int})
        else:
            preprocessing.pop("inv_int", None)

        preprocessing.pop("contrast_enhance_opt", None)
        preprocessing.pop("inv_int_opt", None)

        # remove type and set downsample to last
        preprocessing.pop("image_type", None)

        spatial_preprocessing.update(
            {"downsample": preprocessing.get("downsample")}
        )
        preprocessing.pop("downsample", None)

        # iterate through intensity transformations preprocessing dictionary
        for k, v in preprocessing.items():
            if v is not None:
                print("performing preprocessing: ", k)
                image = v(image)

        image.SetSpacing((self.image_res, self.image_res))

        return image, spatial_preprocessing

    def preprocess_reg_image_spatial(
        self, spatial_preprocessing, imported_transforms=None
    ):
        # spatial preprocessing:
        # imported transforms -> Masking -> rotation -> flipping
        if imported_transforms is not None:
            self.image = transform_2d_image_itkelx(
                self.image, self.image_res, imported_transforms
            )
            if self.mask is not None:
                self.mask = transform_2d_image_itkelx(
                    self.mask, self.image_res, imported_transforms
                )
            self.transforms.append(imported_transforms)
        if spatial_preprocessing is not None:
            if spatial_preprocessing.get("mask_bbox") is not None:
                bbox = spatial_preprocessing["mask_bbox"]
                self.image = self.image[
                    bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[3]
                ]
                if self.mask is not None:
                    self.mask = self.mask[
                        bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[3]
                    ]
                self.transforms.append({"mask_bbox": bbox})

            if spatial_preprocessing.get("rot_cc") is not None:
                rotangle = spatial_preprocessing["rot_cc"]
                if rotangle is not None and rotangle != 0:
                    print(f"rotating counter-clockwise {rotangle}")
                    # self.image.SetSpacing((self.image_res, self.image_res))
                    rot_tform = gen_rigid_tform_rot(
                        self.image, self.image_res, rotangle
                    )
                    self.image = transform_2d_image_itkelx(
                        self.image, [rot_tform]
                    )

                    if self.mask is not None:
                        self.mask = transform_2d_image_itkelx(
                            self.mask, [rot_tform]
                        )

                    self.transforms.append(rot_tform)

            if spatial_preprocessing.get("flip") is not None:
                flip_direction = spatial_preprocessing["flip"]
                if flip_direction != "None" and flip_direction is not None:
                    print(f"flipping image {flip_direction}")

                    flip_tform = gen_aff_tform_flip(
                        self.image, self.image_res, flip_direction
                    )
                    # image.SetSpacing((self.image_res, self.image_res))
                    self.image = transform_2d_image_itkelx(
                        self.image, [flip_tform]
                    )

                    if self.mask is not None:
                        self.mask = transform_2d_image_itkelx(
                            self.mask, [flip_tform]
                        )

                    self.transforms.append(flip_tform)

            downsampling = spatial_preprocessing.get("downsample")

            if downsampling is not None and downsampling > 1:
                print(
                    "performing downsampling by factor: {}".format(
                        downsampling
                    )
                )
                self.image.SetSpacing((self.image_res, self.image_res))
                self.image = sitk.Shrink(
                    self.image, (downsampling, downsampling)
                )

                if self.mask is not None:

                    self.mask.SetSpacing((self.image_res, self.image_res))
                    self.mask = sitk.Shrink(
                        self.mask, (downsampling, downsampling)
                    )

    def sitk_to_itk(self, cast_to_float32=True):
        self.image = sitk_image_to_itk_image(
            self.image, cast_to_float32=cast_to_float32
        )
        if self.mask is not None:
            self.mask = sitk_image_to_itk_image(
                self.mask, cast_to_float32=False
            )
            mask_im_type = itk.Image[itk.UC, 2]
            self.mask = itk.binary_threshold_image_filter(
                self.mask,
                lower_threshold=1,
                inside_value=1,
                ttype=(type(self.mask), mask_im_type),
            )


class TransformRegImage(RegImage):
    """
    Extension of RegImage for performing transformation and outputting to OME-TIFF or OME-zarr

    Parameters
    ----------
    image_name: str
        Name of the image to be used when outputting the data
    image_fp: str
        filepath to the image data
    image_res : float
        Physical spacing (xy resolution) of the image (units are assumed to be the same for each image in experiment but are not
        defined)
        Does not yet support isotropic resolution
    transform_data:list
        list of wsireg elastix transforms
    transforms
    channel_names
    channel_colors

    """

    def __init__(
        self,
        image_name,
        image_fp,
        image_res,
        spatial_preprocessing=None,
        transform_data=None,
        channel_names=None,
        channel_colors=None,
    ):
        """"""
        self.image_name = image_name
        self.image_filepath = image_fp
        self.image_res = image_res
        self.transform_data = transform_data

        im_dims, self.im_dtype, self.reader = get_im_info(self.image_filepath)
        self.im_dims = tuple(im_dims)
        self.is_rgb = guess_rgb(self.im_dims)
        self.mask = None

        self.spatial_prepro = spatial_preprocessing
        self.transforms = None

        self.channel_names = channel_names
        self.channel_colors = channel_colors

        if isinstance(self.transform_data, str) is True:
            self.transform_data = json_to_pmap_dict(self.transform_data)
        if self.transform_data is not None:
            (
                self.composite_transform,
                self.itk_transforms,
            ) = wsireg_transforms_to_itk_composite(self.transform_data)
            self.final_tform = self.itk_transforms[-1]
        else:
            self.final_tform = None
            self.itk_transforms = None
            self.composite_transform = None

    def transform_image(
        self,
        output_dir,
        output_type="ome.tiff",
        tile_size=512,
        write_pyramid=True,
    ):
        """
        Method to transform the image. Will attempt to write layer-by-layer where possible

        Parameters
        ----------
        output_dir:str
            directory where outputs will be saved
        output_type:str
            "ome.zarr" or "ome.tiff"
        tile_size: int
            size of the tiles in the pyramid, default 512
        write_pyramid:bool
            whether to write sub-resolutions of the image

        Returns
        -------
        im_fp:str
            filepath to the saved data
        """
        if output_type.lower() == "ome.zarr" or output_type.lower() == "zarr":
            im_fp = transform_to_ome_zarr(self, output_dir, tile_size)
        if output_type.lower() == "ome.tiff":
            im_fp = transform_to_ome_tiff(
                self,
                output_dir,
                tile_size,
                write_pyramid=write_pyramid,
            )

        return im_fp
