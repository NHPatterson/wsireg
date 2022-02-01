import warnings

import SimpleITK as sitk
from tifffile import TiffFile

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    get_tifffile_info,
    guess_rgb,
    tf_get_largest_series,
    tf_zarr_read_single_ch,
    tifffile_dask_backend,
    tifffile_zarr_backend,
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
        self.image_filepath = image_fp
        self.image_res = image_res
        self.image = None
        self.tf = TiffFile(self.image_filepath)
        self.reader = "tifffile"

        (
            self.im_dims,
            self.im_dtype,
        ) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)

        self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]
        self.mask = self.read_mask(mask)

        self.pre_reg_transforms = pre_reg_transforms

        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self):
        if len(self.tf.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, "
                "the largest series will be read by default"
            )

        im_dims, im_dtype = get_tifffile_info(self.image_filepath)

        return im_dims, im_dtype

    def read_reg_image(self):
        largest_series = tf_get_largest_series(self.image_filepath)

        try:
            reg_image = tifffile_dask_backend(
                self.image_filepath, largest_series, self.preprocessing
            )
        except ValueError:
            reg_image = tifffile_zarr_backend(
                self.image_filepath, largest_series, self.preprocessing
            )

        if (
            self.preprocessing is not None
            and self.preprocessing.get('as_uint8') is True
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
        image = tf_zarr_read_single_ch(
            self.image_filepath, channel_idx, self.is_rgb
        )
        return image
