from pathlib import Path

from wsireg.utils.im_utils import ARRAYLIKE_CLASSES, TIFFFILE_EXTS

from . import CziRegImage  # AICSRegImage,
from . import NumpyRegImage, SitkRegImage, TiffFileRegImage


def reg_image_loader(
    image,
    image_res,
    mask=None,
    pre_reg_transforms=None,
    preprocessing=None,
    channel_names=None,
    channel_colors=None,
):
    if isinstance(image, ARRAYLIKE_CLASSES):
        return NumpyRegImage(
            image,
            image_res,
            mask=mask,
            pre_reg_transforms=pre_reg_transforms,
            preprocessing=preprocessing,
            channel_names=channel_names,
            channel_colors=channel_colors,
            image_filepath=None,
        )

    image_ext = Path(image).suffix.lower()
    if image_ext in TIFFFILE_EXTS:
        reg_image = TiffFileRegImage(
            image,
            image_res,
            mask=mask,
            pre_reg_transforms=pre_reg_transforms,
            preprocessing=preprocessing,
            channel_names=channel_names,
            channel_colors=channel_colors,
        )
    elif image_ext == ".czi":
        reg_image = CziRegImage(
            image,
            image_res,
            mask=mask,
            pre_reg_transforms=pre_reg_transforms,
            preprocessing=preprocessing,
            channel_names=channel_names,
            channel_colors=channel_colors,
        )
    else:
        reg_image = SitkRegImage(
            image,
            image_res,
            mask=mask,
            pre_reg_transforms=pre_reg_transforms,
            preprocessing=preprocessing,
            channel_names=channel_names,
            channel_colors=channel_colors,
        )

    return reg_image
