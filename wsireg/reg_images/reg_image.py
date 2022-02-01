from copy import deepcopy
from typing import Any, Dict, Optional, Union

import itk
import numpy as np
import SimpleITK as sitk

from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.utils.im_utils import (
    compute_mask_to_bbox,
    contrast_enhance,
    sitk_inv_int,
    sitk_max_int_proj,
    transform_plane,
)
from wsireg.utils.tform_utils import (
    gen_aff_tform_flip,
    gen_rig_to_original,
    gen_rigid_tform_rot,
    gen_rigid_translation,
    prepare_wsireg_transform_data,
)


class RegImage:
    """
    Image container class for reading images and preparing them for registration.
    Does two forms of preprocessing:
    Intensity preprocessing, modifying intensity values for best registration
    Spatial preprocessing, rotating and flipping images prior to registration for best alignment

    After preprocessing, images destined for registration should always be a single channel 2D image

    Attributes
    ----------
    image_filepath : str or np.array
        filepath to image to be processed or a numpy array of the image
    image_res : int or float
        Physical spacing (xy resolution) of the image (units are assumed to be the same for each image in experiment but are not
        defined)
        Does not yet support isotropic resolution
    preprocessing : ImagePreproParams
        preprocessing to apply to the image
    transforms: list
        list of elastix transformation data to apply to an image during preprocessing
    mask : str or np.ndarray
        filepath of a mask image to be processed or a numpy array of the mask
        all values => 1 are considered WITHIN the mask
    """

    def __init__(
        self,
        preprocessing: Optional[
            Union[ImagePreproParams, Dict[str, Any]]
        ] = None,
    ):
        self.image = None
        self.reg_image = None
        self.original_size_transform = None
        if preprocessing:
            if isinstance(preprocessing, ImagePreproParams):
                self.preprocessing = preprocessing
            elif isinstance(preprocessing, dict):
                self.preprocessing = ImagePreproParams(**preprocessing)
        else:
            self.preprocessing = ImagePreproParams()

        self.pre_reg_transforms = None
        self.original_size_transform = None
        self.channel_names = None
        self.channel_colors = None

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

    def preprocess_reg_image_intensity(
        self, image, preprocessing: ImagePreproParams
    ):

        if preprocessing.image_type.value == "FL":
            preprocessing.invert_intensity = False
        elif preprocessing.image_type.value == "BF":
            preprocessing.max_int_proj = False
            preprocessing.contrast_enhance = False
            if self.is_rgb:
                preprocessing.invert_intensity = True

        if preprocessing.max_int_proj:
            image = sitk_max_int_proj(image)

        if preprocessing.contrast_enhance:
            image = contrast_enhance(image)

        if preprocessing.invert_intensity:
            image = sitk_inv_int(image)

        if preprocessing.custom_processing:
            for k, v in preprocessing.custom_processing.items():
                print(f"performing preprocessing: {k}")
                image = v(image)

        image.SetSpacing((self.image_res, self.image_res))

        return image

    def preprocess_reg_image_spatial(
        self, image, preprocessing: ImagePreproParams, imported_transforms=None
    ):

        transforms = []
        original_size = image.GetSize()

        if preprocessing.downsampling > 1:
            print(
                "performing downsampling by factor: {}".format(
                    preprocessing.downsampling
                )
            )
            image.SetSpacing((self.image_res, self.image_res))
            image = sitk.Shrink(
                image,
                (preprocessing.downsampling, preprocessing.downsampling),
            )

            if self.mask is not None:
                self.mask.SetSpacing((self.image_res, self.image_res))
                self.mask = sitk.Shrink(
                    self.mask,
                    (
                        preprocessing.downsampling,
                        preprocessing.downsampling,
                    ),
                )

            image_res = image.GetSpacing()[0]
        else:
            image_res = self.image_res

        if float(preprocessing.rot_cc) != 0.0:
            print(f"rotating counter-clockwise {preprocessing.rot_cc}")
            rot_tform = gen_rigid_tform_rot(
                image, image_res, preprocessing.rot_cc
            )
            (
                composite_transform,
                _,
                final_tform,
            ) = prepare_wsireg_transform_data({"initial": [rot_tform]})

            image = transform_plane(image, final_tform, composite_transform)

            if self.mask is not None:
                self.mask.SetSpacing((image_res, image_res))
                self.mask = transform_plane(
                    self.mask, final_tform, composite_transform
                )
            transforms.append(rot_tform)

        if preprocessing.flip:
            print(f"flipping image {preprocessing.flip.value}")

            flip_tform = gen_aff_tform_flip(
                image, image_res, preprocessing.flip.value
            )

            (
                composite_transform,
                _,
                final_tform,
            ) = prepare_wsireg_transform_data({"initial": [flip_tform]})

            image = transform_plane(image, final_tform, composite_transform)

            if self.mask is not None:
                self.mask.SetSpacing((image_res, image_res))
                self.mask = transform_plane(
                    self.mask, final_tform, composite_transform
                )

            transforms.append(flip_tform)

        if self.mask and preprocessing.crop_to_mask_bbox:
            print("computing mask bounding box")
            if preprocessing.mask_bbox is None:
                mask_bbox = compute_mask_to_bbox(self.mask)
                preprocessing.mask_bbox = mask_bbox

        if preprocessing.mask_bbox:

            print("cropping to mask")
            translation_transform = gen_rigid_translation(
                image,
                image_res,
                preprocessing.mask_bbox.X,
                preprocessing.mask_bbox.Y,
                preprocessing.mask_bbox.WIDTH,
                preprocessing.mask_bbox.HEIGHT,
            )

            (
                composite_transform,
                _,
                final_tform,
            ) = prepare_wsireg_transform_data(
                {"initial": [translation_transform]}
            )

            image = transform_plane(image, final_tform, composite_transform)

            self.original_size_transform = gen_rig_to_original(
                original_size, deepcopy(translation_transform)
            )

            if self.mask is not None:
                self.mask.SetSpacing((image_res, image_res))
                self.mask = transform_plane(
                    self.mask, final_tform, composite_transform
                )
            transforms.append(translation_transform)

        return image, transforms

    def preprocess_image(self, reg_image):

        reg_image = self.preprocess_reg_image_intensity(
            reg_image, self.preprocessing
        )

        if reg_image.GetDepth() >= 1:
            raise ValueError(
                "preprocessing did not result in a single image plane\n"
                "multi-channel or 3D image return"
            )

        if reg_image.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError(
                "preprocessing did not result in a single image plane\n"
                "multi-component / RGB(A) image returned"
            )

        reg_image, pre_reg_transforms = self.preprocess_reg_image_spatial(
            reg_image, self.preprocessing, self.pre_reg_transforms
        )

        if len(pre_reg_transforms) > 0:
            self.pre_reg_transforms = pre_reg_transforms

        self.reg_image = reg_image

    def reg_image_sitk_to_itk(self, cast_to_float32=True):
        origin = self.reg_image.GetOrigin()
        spacing = self.reg_image.GetSpacing()
        # direction = image.GetDirection()
        is_vector = self.reg_image.GetNumberOfComponentsPerPixel() > 1
        if cast_to_float32 is True:
            self.reg_image = sitk.Cast(self.reg_image, sitk.sitkFloat32)
            self.reg_image = sitk.GetArrayFromImage(self.reg_image)
        else:
            self.reg_image = sitk.GetArrayFromImage(self.reg_image)

        self.reg_image = itk.GetImageFromArray(
            self.reg_image, is_vector=is_vector
        )
        self.reg_image.SetOrigin(origin)
        self.reg_image.SetSpacing(spacing)

        if self.mask is not None:
            origin = self.mask.GetOrigin()
            spacing = self.mask.GetSpacing()
            # direction = image.GetDirection()
            is_vector = self.mask.GetNumberOfComponentsPerPixel() > 1
            if cast_to_float32 is True:
                self.mask = sitk.Cast(self.mask, sitk.sitkFloat32)
                self.mask = sitk.GetArrayFromImage(self.mask)
            else:
                self.mask = sitk.GetArrayFromImage(self.mask)

            self.mask = itk.GetImageFromArray(self.mask, is_vector=is_vector)
            self.mask.SetOrigin(origin)
            self.mask.SetSpacing(spacing)

            mask_im_type = itk.Image[itk.UC, 2]
            self.mask = itk.binary_threshold_image_filter(
                self.mask,
                lower_threshold=1,
                inside_value=1,
                ttype=(type(self.mask), mask_im_type),
            )
