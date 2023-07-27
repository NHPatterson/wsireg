import warnings

import SimpleITK as sitk

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    ensure_dask_array,
    get_sitk_image_info,
    guess_rgb,
)


class SitkRegImage(RegImage):
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
        super(SitkRegImage, self).__init__(preprocessing)
        self._path = image
        self._image_res = image_res

        self.reader = "sitk"

        (
            self._shape,
            self._im_dtype,
        ) = self._get_image_info()

        self._is_rgb = guess_rgb(self._shape)

        self._n_ch = self._shape[2] if self.is_rgb else self._shape[0]

        if mask:
            self._mask = self.read_mask(mask)

        self.pre_reg_transforms = pre_reg_transforms

        self._channel_names = channel_names
        self._channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self):
        im_dims, im_dtype = get_sitk_image_info(self._path)

        im_dims = (int(im_dims[0]), int(im_dims[1]), int(im_dims[2]))

        return im_dims, im_dtype

    def read_reg_image(self):
        """
        Read and preprocess the image for registration.
        """
        reg_image = sitk.ReadImage(self._path)

        if self.preprocessing.as_uint8 is True and reg_image.GetPixelID() != 1:
            reg_image = sitk.Cast(
                sitk.RescaleIntensity(reg_image), sitk.sitkUInt8
            )

        self.preprocess_image(reg_image)

    def _read_full_image(self):
        self._dask_image = ensure_dask_array(
            sitk.GetArrayFromImage(sitk.ReadImage(self._path))
        )

        rechunk_size = (
            (2048, 2048, self.n_ch) if self.is_rgb else (self.n_ch, 2048, 2048)
        )
        self._dask_image = self._dask_image.rechunk(rechunk_size)

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
