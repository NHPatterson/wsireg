from copy import deepcopy
import numpy as np
import SimpleITK as sitk
import itk
from wsireg.utils.im_utils import (
    contrast_enhance,
    sitk_inv_int,
    transform_to_ome_tiff,
    transform_to_ome_zarr,
    compute_mask_to_bbox,
)
from wsireg.utils.tform_utils import (
    transform_2d_image_itkelx,
    gen_aff_tform_flip,
    gen_rigid_tform_rot,
    gen_rigid_translation,
    gen_rig_to_original,
    prepare_wsireg_transform_data,
)
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

    def __init__(self):
        self.original_size_transform = None
        return

    def read_mask(self, mask):
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = sitk.GetImageFromArray(mask)
            elif isinstance(mask, str):
                mask = sitk.ReadImage(mask)
            elif isinstance(mask, sitk.Image):
                mask = mask

            mask.SetSpacing((self.image_res, self.image_res))

        return mask

    def preprocess_reg_image_intensity(self, image, preprocessing):
        # if isinstance(image, Path):
        #     image = read_image(str(image), preprocessing=preprocessing)
        # else:
        #     image = prepare_np_image(image, preprocessing=preprocessing)

        # separate spatial preprocessing from intensity preprocessing
        spatial_preprocessing = {}
        for spat_key in [
            "mask_to_bbox",
            "mask_bbox",
            "rot_cc",
            "flip",
            "use_mask",
        ]:
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

        if preprocessing.get("inv_int_opt") is True or self.is_rgb is True:
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
        self, image, spatial_preprocessing, imported_transforms=None
    ):
        # spatial preprocessing:
        # imported transforms -> Masking -> rotation -> flipping
        transforms = []
        # if imported_transforms is not None:
        #     image = transform_2d_image_itkelx(
        #         image, self.image_res, imported_transforms
        #     )
        #     if self.mask is not None:
        #         self.mask = transform_2d_image_itkelx(
        #             self.mask, self.image_res, imported_transforms
        #         )
        #     transforms.append(imported_transforms)
        original_size = image.GetSize()
        if spatial_preprocessing is not None:
            if (
                self.mask is not None
                and spatial_preprocessing.get("mask_to_bbox") is not None
            ):
                if spatial_preprocessing.get("mask_to_bbox") is True:
                    print("computing mask bounding box")
                    mask_bbox = compute_mask_to_bbox(self.mask)
                    spatial_preprocessing.update({"mask_bbox": mask_bbox})

            if spatial_preprocessing.get("mask_bbox") is not None:
                bbox = spatial_preprocessing["mask_bbox"]
                print("cropping to mask")
                translation_transform = gen_rigid_translation(
                    image, self.image_res, bbox[0], bbox[1], bbox[2], bbox[3]
                )
                image = transform_2d_image_itkelx(
                    image, [translation_transform]
                )

                self.original_size_transform = gen_rig_to_original(
                    original_size, deepcopy(translation_transform)
                )

                if self.mask is not None:
                    self.mask.SetSpacing((self.image_res, self.image_res))
                    self.mask = transform_2d_image_itkelx(
                        self.mask, [translation_transform]
                    )
                transforms.append(translation_transform)

            if spatial_preprocessing.get("rot_cc") is not None:
                rotangle = spatial_preprocessing["rot_cc"]
                if rotangle is not None and rotangle != 0:
                    print(f"rotating counter-clockwise {rotangle}")
                    rot_tform = gen_rigid_tform_rot(
                        image, self.image_res, rotangle
                    )
                    image = transform_2d_image_itkelx(image, [rot_tform])

                    if self.mask is not None:
                        self.mask = transform_2d_image_itkelx(
                            self.mask, [rot_tform]
                        )

                    transforms.append(rot_tform)

            if spatial_preprocessing.get("flip") is not None:
                flip_direction = spatial_preprocessing["flip"]
                if flip_direction != "None" and flip_direction is not None:
                    print(f"flipping image {flip_direction}")

                    flip_tform = gen_aff_tform_flip(
                        image, self.image_res, flip_direction
                    )

                    image = transform_2d_image_itkelx(image, [flip_tform])

                    if self.mask is not None:
                        self.mask = transform_2d_image_itkelx(
                            self.mask, [flip_tform]
                        )

                    transforms.append(flip_tform)

            downsampling = spatial_preprocessing.get("downsample")

            if downsampling is not None and downsampling > 1:
                print(
                    "performing downsampling by factor: {}".format(
                        downsampling
                    )
                )
                image.SetSpacing((self.image_res, self.image_res))
                image = sitk.Shrink(image, (downsampling, downsampling))

                if self.mask is not None:

                    self.mask.SetSpacing((self.image_res, self.image_res))
                    self.mask = sitk.Shrink(
                        self.mask, (downsampling, downsampling)
                    )
        if len(transforms) == 0:
            transforms = None
        if (
            spatial_preprocessing.get("use_mask") is not None
            and spatial_preprocessing.get("use_mask") is False
        ):
            self.mask = None
        return image, transforms

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

    def transform_image(
        self,
        image_name,
        transform_data,
        file_writer="ome.tiff",
        output_dir="",
        **transformation_opts,
    ):
        if transform_data is not None:
            (
                itk_composite,
                itk_transforms,
                final_transform,
            ) = prepare_wsireg_transform_data(transform_data)
        else:
            itk_composite, itk_transforms, final_transform = None, None, None

        itk_transforms
        if file_writer.lower() == "ome.zarr" or file_writer.lower() == "zarr":
            im_fp = transform_to_ome_zarr(
                self, output_dir, **transformation_opts
            )
        if file_writer.lower() == "ome.tiff":
            im_fp = transform_to_ome_tiff(
                self,
                image_name,
                output_dir,
                final_transform,
                itk_composite,
                **transformation_opts,
            )
        return im_fp
