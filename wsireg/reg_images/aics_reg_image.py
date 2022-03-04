import warnings

import dask.array as da
import numpy as np
import SimpleITK as sitk
from aicsimageio import AICSImage

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    ensure_dask_array,
    guess_rgb,
    preprocess_dask_array,
)


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
        super(AICSRegImage, self).__init__(preprocessing)
        self._path = image
        self._image_res = image_res
        self.reader = "aics"

        self.aics_image = AICSImage(self._path)

        dask_image = da.queeze(ensure_dask_array(self.aics_image.dask_data))

        self._dask_image = (
            dask_image.reshape(1, *dask_image.shape)
            if len(dask_image.shape) == 2
            else dask_image
        )

        (
            self._shape,
            self.im_dtype,
        ) = self._get_image_info()

        self._is_rgb = guess_rgb(self._shape)

        self._n_ch = self._shape[2] if self._is_rgb else self._shape[0]

        if mask:
            self._mask = self.read_mask(mask)

        self.pre_reg_transforms = pre_reg_transforms

        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self):
        im_dims = self._dask_image.shape
        im_dtype = self._dask_image.dtype

        return im_dims, im_dtype

    def read_reg_image(self):
        """
        Read and preprocess the image for registration.
        """
        reg_image = self._dask_image
        reg_image = preprocess_dask_array(reg_image, self.preprocessing)

        if (
            self.preprocessing is not None
            and self.preprocessing.as_uint8 is True
            and reg_image.GetPixelID() != sitk.sitkUInt8
        ):
            reg_image = sitk.RescaleIntensity(reg_image)
            reg_image = sitk.Cast(reg_image, sitk.sitkUInt8)

        self.preprocess_image(reg_image)

    def read_single_channel(self, channel_idx: int) -> np.ndarray:
        """
        Read in a single channel for transformation by plane.
        Parameters
        ----------
        channel_idx: int
            Index of the channel to be read

        Returns
        -------
        image: np.ndarray
            Numpy array of the selected channel to be read
        """
        if channel_idx > (self.n_ch - 1):
            warnings.warn(
                "channel_idx exceeds number of channels, reading channel at channel_idx == 0"
            )
            channel_idx = 0

        if self._is_rgb:
            image = self._dask_image[:, :, channel_idx].compute()
        else:
            image = self._dask_image[channel_idx, :, :].compute()

        return image
