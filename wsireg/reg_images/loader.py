from pathlib import Path
import numpy as np
from tifffile import TiffFile
from wsireg.reg_images import (
    CziRegImage,
    SitkRegImage,
    TiffFileRegImage,
    NumpyRegImage,
    OmeTiffRegImage
)
from wsireg.utils.im_utils import TIFFFILE_EXTS


def reg_image_loader(
    image_fp,
    image_res,
    mask=None,
    pre_reg_transforms=None,
    preprocessing=None,
    channel_names=None,
    channel_colors=None,
):
    if isinstance(image_fp, np.ndarray):
        return NumpyRegImage(
            image_fp,
            image_res,
            mask,
            pre_reg_transforms,
            preprocessing,
            channel_names,
            channel_colors,
        )

    image_ext = Path(image_fp).suffix
    if image_ext in TIFFFILE_EXTS:
        if TiffFile(image_fp).is_ome:
            reg_image = OmeTiffRegImage(
                image_fp,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
        else:
            reg_image = TiffFileRegImage(
                image_fp,
                image_res,
                mask,
                pre_reg_transforms,
                preprocessing,
                channel_names,
                channel_colors,
            )
    elif image_ext == ".czi":
        reg_image = CziRegImage(
            image_fp,
            image_res,
            mask,
            pre_reg_transforms,
            preprocessing,
            channel_names,
            channel_colors,
        )
    else:
        reg_image = SitkRegImage(
            image_fp,
            image_res,
            mask,
            pre_reg_transforms,
            preprocessing,
            channel_names,
            channel_colors,
        )

    return reg_image
