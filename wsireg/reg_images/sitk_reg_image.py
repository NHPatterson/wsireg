import numpy as np
import SimpleITK as sitk
from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    std_prepro,
    guess_rgb,
    get_sitk_image_info,
    sitk_vect_to_gs,
)


class SitkRegImage(RegImage):
    def __init__(
        self,
        image_fp,
        image_res,
        mask=None,
        pre_reg_transforms=None,
        preprocessing=None,
        channel_names=None,
        channel_colors=None,
    ):
        self.image_filepath = image_fp
        self.image_res = image_res
        self.image = None
        self.reader = "sitk"

        (
            self.im_dims,
            self.im_dtype,
        ) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)

        self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]

        self.mask = self.read_mask(mask)

        if preprocessing is None:
            self.preprocessing = std_prepro()
        else:
            self.preprocessing = std_prepro()
            self.preprocessing.update(preprocessing)

        self.pre_reg_transforms = pre_reg_transforms

        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self):

        im_dims, im_dtype = get_sitk_image_info(self.image_filepath)

        return im_dims, im_dtype

    def read_reg_image(self):
        image = sitk.ReadImage(self.image_filepath)

        if image.GetNumberOfComponentsPerPixel() >= 3:
            if self.preprocessing is not None:
                image = sitk_vect_to_gs(image)

        if (
            self.preprocessing.get("as_uint8") is True
            and image.GetPixelID() != 1
        ):
            image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)

        if (
            self.preprocessing.get("ch_indices") is not None
            and image.GetDepth() > 0
        ):
            chs = np.asarray(self.preprocessing.get('ch_indices'))
            image = image[:, :, chs]

        image, spatial_preprocessing = self.preprocess_reg_image_intensity(
            image, self.preprocessing
        )

        if image.GetDepth() >= 1:
            raise ValueError(
                "preprocessing did not result in a single image plane\n"
                "multi-channel or 3D image return"
            )

        if image.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError(
                "preprocessing did not result in a single image plane\n"
                "multi-component / RGB(A) image returned"
            )

        if (
            len(spatial_preprocessing) > 0
            or self.pre_reg_transforms is not None
        ):
            (
                self.image,
                self.pre_reg_transforms,
            ) = self.preprocess_reg_image_spatial(
                image, spatial_preprocessing, self.pre_reg_transforms
            )
        else:
            self.image = image
            self.pre_reg_transforms = None
