from pathlib import Path
from tifffile import TiffFile
from wsireg.reg_images import (
    CziRegImage,
    SitkRegImage,
    NumpyRegImage,
    AICSRegImage,
)
from wsireg.utils.im_utils import (
    TIFFFILE_EXTS,
    ARRAYLIKE_CLASSES,
    tifffile_to_arraylike,
    ome_tifffile_to_arraylike,
)


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
            image = ome_tifffile_to_arraylike(image)
            reg_image = NumpyRegImage(
                image,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
        else:
            image = tifffile_to_arraylike(image)
            reg_image = NumpyRegImage(
                image,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
    elif image_ext == ".czi":
        try:
            reg_image = AICSRegImage(
                image,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
        except ValueError:
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
