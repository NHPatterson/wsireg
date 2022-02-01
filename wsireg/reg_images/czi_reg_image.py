import numpy as np
import SimpleITK as sitk

from wsireg.reg_images import RegImage
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
        self.image_filepath = image
        self.image_res = image_res
        self.image = None
        self.czi = CziRegImageReader(self.image_filepath)
        self.reader = "czi"

        (
            self.ch_dim_idx,
            self.y_dim_idx,
            self.x_dim_idx,
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

        return ch_dim_idx, y_dim_idx, x_dim_idx, im_dims, im_dtype

    def read_reg_image(self):
        scene_idx = self.czi.axes.index('S')

        if self.czi.shape[scene_idx] > 1:
            raise ValueError('multi scene czis not allowed at this time')
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
        if self.is_rgb is False:
            image = self.czi.sub_asarray(
                channel_idx=[channel_idx],
            )
        else:
            image = self.czi.sub_asarray_rgb(
                channel_idx=[channel_idx], greyscale=False
            )

        return image
