import numpy as np
import SimpleITK as sitk

from wsireg.reg_images.reg_image import RegImage
from wsireg.utils.im_utils import (
    ensure_dask_array,
    get_sitk_image_info,
    guess_rgb,
    sitk_vect_to_gs,
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
        self.image_filepath = image
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

        self.reg_image = None
        self.mask = self.read_mask(mask)

        self.pre_reg_transforms = pre_reg_transforms

        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.original_size_transform = None

    def _get_image_info(self):

        im_dims, im_dtype = get_sitk_image_info(self.image_filepath)

        return im_dims, im_dtype

    def read_reg_image(self):
        reg_image = sitk.ReadImage(self.image_filepath)

        if reg_image.GetNumberOfComponentsPerPixel() >= 3:
            if self.preprocessing is not None:
                reg_image = sitk_vect_to_gs(reg_image)

        if (
            self.preprocessing.get("as_uint8") is True
            and reg_image.GetPixelID() != 1
        ):
            reg_image = sitk.Cast(
                sitk.RescaleIntensity(reg_image), sitk.sitkUInt8
            )

        if (
            self.preprocessing.get("ch_indices") is not None
            and reg_image.GetDepth() > 0
        ):
            chs = np.asarray(self.preprocessing.get('ch_indices'))
            reg_image = reg_image[:, :, chs]

        self.preprocess_image(reg_image)

    def read_full_image(self):
        self.image = ensure_dask_array(
            sitk.GetArrayFromImage(sitk.ReadImage(self.image_filepath))
        )
