import warnings
from typing import Tuple

import dask.array as da
import numpy as np
import SimpleITK as sitk

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import CziRegImageReader, guess_rgb


class CziRegImage(RegImage):
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
        super(CziRegImage, self).__init__(preprocessing)
        self._path = image
        self._image_res = image_res

        self.czi = CziRegImageReader(self._path)
        self.reader = "czi"

        scene_idx = self.czi.axes.index('S')

        if self.czi.shape[scene_idx] > 1:
            raise ValueError('multi scene czis not allowed at this time')

        (
            self._shape,
            self._im_dtype,
        ) = self._get_image_info()

        self._is_rgb = guess_rgb(self._shape)
        self._n_ch = self._shape[2] if self.is_rgb else self._shape[0]

        self._dask_image = self._prepare_dask_image()

        if mask:
            self._mask = self.read_mask(mask)

        self.pre_reg_transforms = pre_reg_transforms

        self._channel_names = channel_names
        self._channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self):
        # if RGB need to get 0
        if self.czi.shape[-1] > 1:
            ch_dim_idx = self.czi.axes.index('0')
        else:
            ch_dim_idx = self.czi.axes.index('C')
        y_dim_idx = self.czi.axes.index('Y')
        x_dim_idx = self.czi.axes.index('X')
        if self.czi.shape[-1] > 1:
            im_dims = np.array(self.czi.shape)[
                [y_dim_idx, x_dim_idx, ch_dim_idx]
            ]
        else:
            im_dims = np.array(self.czi.shape)[
                [ch_dim_idx, y_dim_idx, x_dim_idx]
            ]

        im_dtype = self.czi.dtype

        im_dims = (int(im_dims[0]), int(im_dims[1]), int(im_dims[2]))

        return im_dims, im_dtype

    def _prepare_dask_image(self) -> da.Array:
        ch_dim = self._shape[1:] if not self._is_rgb else self._shape[:2]
        chunks = ((1,) * self._n_ch, (ch_dim[0],), (ch_dim[1],))
        dask_image = da.map_blocks(
            self._czi_read_single_channel,
            chunks=chunks,
            dtype=self.im_dtype,
            meta=np.array((), dtype=self._im_dtype),
        )
        return dask_image

    def _czi_read_single_channel(self, block_id: Tuple[int, ...]):
        channel_idx = block_id[0]
        if self.is_rgb is False:
            image = self.czi.sub_asarray(
                channel_idx=[channel_idx],
            )
        else:
            image = self.czi.sub_asarray_rgb(
                channel_idx=[channel_idx], greyscale=False
            )

        return np.expand_dims(np.squeeze(image), axis=0)

    def read_reg_image(self):
        """
        Read and preprocess the image for registration.
        For the Zeiss CZI reader, this involves grayscaling RGB on read
        or reading only a subset of the channel images.
        """
        if self.is_rgb:
            reg_image = self.czi.sub_asarray_rgb(greyscale=True)
        else:
            reg_image = self.czi.sub_asarray(
                channel_idx=self.preprocessing.ch_indices,
                as_uint8=self.preprocessing.as_uint8,
            )

        reg_image = np.squeeze(reg_image)
        reg_image = sitk.GetImageFromArray(reg_image)

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

        image = self._dask_image[channel_idx, :, :].compute()

        return image
