import warnings
import SimpleITK as sitk
import numpy as np
import dask.array as da
from wsireg.reg_images import RegImage
from wsireg.utils.im_utils import (
    std_prepro,
    guess_rgb,
    read_preprocess_array,
    ensure_dask_array,
)
from aicsimageio import AICSImage


class AICSRegImage(RegImage):
    def __init__(
        self,
        image,
        image_res,
        mask=None,
        pre_reg_transforms=None,
        preprocessing=None,
        channel_names=None,
        channel_colors=None,
    ):

        self.image_filepath = image
        self.image_res = image_res
        self.reader = "aics"
        self.aics_image = AICSImage(self.image_filepath)
        self.image = ensure_dask_array(self.aics_image.dask_data)

        self.image = da.squeeze(self.image)

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
        im_dims = self.image.shape
        if len(im_dims) == 2:
            im_dims = np.concatenate([[1], im_dims])
        im_dtype = self.image.dtype
        return im_dims, im_dtype

    def read_reg_image(self):

        image = read_preprocess_array(
            self.image, preprocessing=self.preprocessing, force_rgb=self.is_rgb
        )

        if (
            self.preprocessing is not None
            and self.preprocessing.get('as_uint8') is True
            and image.GetPixelID() != sitk.sitkUInt8
        ):
            image = sitk.RescaleIntensity(image)
            image = sitk.Cast(image, sitk.sitkUInt8)

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

    def read_single_channel(self, channel_idx: int):
        if channel_idx > (self.n_ch - 1):
            warnings.warn(
                "channel_idx exceeds number of channels, reading channel at channel_idx == 0"
            )
            channel_idx = 0
        if self.n_ch > 1:
            if self.is_rgb:
                image = self.image[:, :, channel_idx]
            else:
                image = self.image[channel_idx, :, :]
        else:
            image = self.image

        return np.asarray(image)
