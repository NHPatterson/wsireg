import warnings

import SimpleITK as sitk
from tifffile import TiffFile, xml2dict

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    get_tifffile_info,
    guess_rgb,
    std_prepro,
    tf_get_largest_series,
    tf_zarr_read_single_ch,
    tifffile_dask_backend,
    tifffile_zarr_backend,
)


class OmeTiffRegImage(RegImage):
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
        super(OmeTiffRegImage, self).__init__(preprocessing)
        self.image_filepath = image_fp
        self.image_res = image_res
        self.image = None
        self.tf = TiffFile(self.image_filepath)
        self.reader = "tifffile"

        self.ome_metadata = xml2dict(self.tf.ome_metadata)

        (
            self.im_dims,
            self.im_dtype,
        ) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)

        # self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]
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
        if len(self.tf.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, "
                "the largest series will be read by default"
            )
            largest_series = tf_get_largest_series(self.image_filepath)
            series_metadata = self.ome_metadata.get("OME").get("Image")[
                largest_series
            ]
        else:
            series_metadata = self.ome_metadata.get("OME").get("Image")

        im_dims, im_dtype = get_tifffile_info(self.image_filepath)

        if isinstance(series_metadata.get("Pixels").get("Channel"), list):
            samples_per_pixel = (
                series_metadata.get("Pixels")
                .get("Channel")[0]
                .get("SamplesPerPixel")
            )
        else:
            samples_per_pixel = (
                series_metadata.get("Pixels")
                .get("Channel")
                .get("SamplesPerPixel")
            )

        is_rgb = guess_rgb(im_dims)
        self.n_ch = im_dims[2] if is_rgb else im_dims[0]

        if is_rgb is False and samples_per_pixel >= 3:
            self.is_rgb = True
            self.is_rgb_interleaved = False
        elif is_rgb is True and samples_per_pixel >= 3:
            self.is_rgb = True
            self.is_rgb_interleaved = True
        else:
            self.is_rgb = False
            self.is_rgb_interleaved = False

        return im_dims, im_dtype

    def read_reg_image(self):
        largest_series = tf_get_largest_series(self.image_filepath)

        if self.is_rgb is True and self.is_rgb_interleaved is False:
            force_rgb = True
        else:
            force_rgb = None

        try:
            reg_image = tifffile_dask_backend(
                self.image_filepath,
                largest_series,
                self.preprocessing,
                force_rgb=force_rgb,
            )
        except ValueError:
            reg_image = tifffile_zarr_backend(
                self.image_filepath,
                largest_series,
                self.preprocessing,
                force_rgb=force_rgb,
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
            self.image_filepath,
            channel_idx,
            self.is_rgb,
            is_rgb_interleaved=self.is_rgb_interleaved,
        )
        return image
