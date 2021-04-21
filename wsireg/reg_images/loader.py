from pathlib import Path
from tifffile import TiffFile
from wsireg.reg_images import (
    CziRegImage,
    SitkRegImage,
    TiffFileRegImage,
    NumpyRegImage,
    OmeTiffRegImage,
)
from wsireg.utils.im_utils import TIFFFILE_EXTS, ARRAYLIKE_CLASSES


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
            mask,
            pre_reg_transforms,
            preprocessing,
            channel_names,
            channel_colors,
        )

    image_ext = Path(image).suffix
    if image_ext in TIFFFILE_EXTS:
        if TiffFile(image).is_ome:
            reg_image = OmeTiffRegImage(
                image,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
        else:
            reg_image = TiffFileRegImage(
                image,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
    elif image_ext == ".czi":
        reg_image = CziRegImage(
            image,
            image_res,
            mask,
            pre_reg_transforms,
            preprocessing,
            channel_names,
            channel_colors,
        )
    else:
        reg_image = SitkRegImage(
            image,
            image_res,
            mask,
            pre_reg_transforms,
            preprocessing,
            channel_names,
            channel_colors,
        )

    return reg_image
