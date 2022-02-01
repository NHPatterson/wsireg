import warnings

import dask.array as da
import numpy as np
import SimpleITK as sitk

from wsireg.reg_images import RegImage
from wsireg.utils.im_utils import (
    ensure_dask_array,
    guess_rgb,
    read_preprocess_array,
)


class NumpyRegImage(RegImage):
    def __init__(
        self,
        image,
        image_res,
        mask=None,
        pre_reg_transforms=None,
        preprocessing=None,
        channel_names=None,
        channel_colors=None,
        image_filepath=None,
    ):
        super(NumpyRegImage, self).__init__(preprocessing)
        self.image_filepath = image_filepath
        self.image_res = image_res
        self.reader = "numpy"

        self.image = ensure_dask_array(image)
        self.image = da.squeeze(self.image)

        (
            self.im_dims,
            self.im_dtype,
        ) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)

        self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]

        self.reg_image = None
        self.mask = self.read_mask(mask)

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

        reg_image = read_preprocess_array(
            self.image, preprocessing=self.preprocessing, force_rgb=self.is_rgb
        )

        if (
            self.preprocessing
            and self.preprocessing.as_uint8
            and reg_image.GetPixelID() != sitk.sitkUInt8
        ):
            reg_image = sitk.RescaleIntensity(reg_image)
            reg_image = sitk.Cast(reg_image, sitk.sitkUInt8)

        self.preprocess_image(reg_image)

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
