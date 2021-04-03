import warnings
import SimpleITK as sitk
import numpy as np
from wsireg.reg_image import RegImage
from wsireg.utils.im_utils import (
    std_prepro,
    guess_rgb,
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
    ):
        self.image_filepath = "numpy"
        self.image_res = image_res
        self.image = image
        self.reader = "numpy"

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
        im_dims = np.squeeze(self.image.shape)
        if len(im_dims) == 2:
            im_dims = np.concatenate([[1], im_dims])
        im_dtype = self.image.dtype

        return im_dims, im_dtype

    def read_reg_image(self):

        if self.is_rgb == True:  # noqa: E712
            self.image = np.dot(
                self.image[..., :3], [0.299, 0.587, 0.114]
            ).astype(np.uint8)

        if self.preprocessing.get("ch_indices") is not None:
            chs = np.asarray(self.preprocessing.get('ch_indices'))
            if self.is_rgb == False:  # noqa: E712
                self.image = self.image[chs, :, :]

        self.image = np.squeeze(self.image)

        image = sitk.GetImageFromArray(self.image)

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

        return image
