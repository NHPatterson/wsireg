import warnings
from typing import List, Tuple

import dask.array as da
import numpy as np
import SimpleITK as sitk
from ome_types import from_xml
from tifffile import TiffFile

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    get_tifffile_info,
    guess_rgb,
    preprocess_dask_array,
    tifffile_to_dask,
)


class TiffFileRegImage(RegImage):
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
        super(TiffFileRegImage, self).__init__(preprocessing)
        self._path = image_fp
        self._image_res = image_res
        self.tf = TiffFile(self._path)
        self.reader = "tifffile"

        (
            self._shape,
            self._im_dtype,
            self.largest_series,
        ) = self._get_image_info()

        self._get_dim_info()

        self._dask_image = self._get_dask_image()

        if mask:
            self._mask = self.read_mask(mask)

        self.pre_reg_transforms = pre_reg_transforms

        self._channel_names = channel_names
        self._channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self) -> Tuple[Tuple[int, int, int], np.dtype, int]:
        if len(self.tf.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, "
                "the largest series will be read by default"
            )

        im_dims, im_dtype, largest_series = get_tifffile_info(self._path)

        im_dims = (int(im_dims[0]), int(im_dims[1]), int(im_dims[2]))

        return im_dims, im_dtype, largest_series

    def _get_dim_info(self) -> None:
        if self._shape:
            if self.tf.ome_metadata:
                self.ome_metadata = from_xml(self.tf.ome_metadata)
                spp = (
                    self.ome_metadata.images[self.largest_series]
                    .pixels.channels[0]
                    .samples_per_pixel
                )
                interleaved = self.ome_metadata.images[
                    self.largest_series
                ].pixels.interleaved

                if spp and spp > 1:
                    self._is_rgb = True
                else:
                    self._is_rgb = False

                if guess_rgb(self._shape) is False:
                    self._channel_axis = 0
                    self._is_interleaved = False
                elif interleaved and guess_rgb(self._shape):
                    self._is_interleaved = True
                    self._channel_axis = len(self._shape) - 1

            else:
                self._is_rgb = guess_rgb(self._shape)
                self._is_interleaved = self._is_rgb
                if self._is_rgb:
                    self._channel_axis = len(self._shape) - 1
                else:
                    self._channel_axis = 0

            self._n_ch = self._shape[self._channel_axis]

    def _get_dask_image(self) -> List[da.Array]:
        dask_image = tifffile_to_dask(self._path, self.largest_series, level=0)
        dask_image = (
            dask_image.reshape(1, *dask_image.shape)
            if len(dask_image.shape) == 2
            else dask_image
        )

        if self._is_rgb and not self._is_interleaved:
            dask_image = da.rollaxis(dask_image, 0, 3)

        return dask_image

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

    def read_single_channel(self, channel_idx: int):
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
