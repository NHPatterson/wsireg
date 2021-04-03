import numpy as np
import SimpleITK as sitk
from wsireg.reg_image import RegImage
from wsireg.utils.im_utils import (
    std_prepro,
    guess_rgb,
    CziRegImageReader,
)


class CziRegImage(RegImage):
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
        self.image_filepath = image_fp
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
        if self.is_rgb is False:
            if self.preprocessing is None:
                image = self.czi.asarray()
            else:
                image = self.czi.sub_asarray(
                    channel_idx=self.preprocessing['ch_indices'],
                    as_uint8=self.preprocessing['as_uint8'],
                )
        else:
            if self.preprocessing is None:
                image = self.czi.asarray()
            else:
                image = self.czi.sub_asarray_rgb(greyscale=True)

        image = np.squeeze(image)
        image = sitk.GetImageFromArray(image)
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
        if self.is_rgb is False:
            image = self.czi.sub_asarray(
                channel_idx=[channel_idx],
            )
        else:
            image = self.czi.sub_asarray_rgb(
                channel_idx=[channel_idx], greyscale=False
            )

        return image
