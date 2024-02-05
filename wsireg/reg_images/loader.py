from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import dask.array as da
import zarr
from wsireg.utils.im_utils import ARRAYLIKE_CLASSES, TIFFFILE_EXTS
from wsireg.parameter_maps.preprocessing import ImagePreproParams
from . import CziRegImage
from . import NumpyRegImage, SitkRegImage, TiffFileRegImage


def reg_image_loader(
    image: Union[np.ndarray, da.Array, zarr.Array, str, Path],
    image_res: Union[int, float],
    mask: Optional[Union[np.ndarray, str, Path]] = None,
    pre_reg_transforms: Optional[dict] = None,
    preprocessing: Optional[ImagePreproParams] = None,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[str]] = None,
) -> Union[TiffFileRegImage, SitkRegImage, NumpyRegImage, CziRegImage]:
    """
    Convenience function to read in images. Determines the correct reader.

    Parameters
    ----------
    image : str, array-like
        file path to the image to be read or an array like image such as
        a numpy, dask or zarr array
    image_res : float
        spatial resolution of image in units per px (i.e. 0.9 um / px)
    mask: Union[str, Path, np.ndarray]
        path to binary mask (>0 is in) image for registration and/or cropping or a geoJSON with shapes
        that will be processed to a binary mask
    pre_reg_transforms: dict
        Pre-computed transforms to be applied to the image prior to registration
    preprocessing: ImagePreproParams
        preprocessing parameters for the modality for registration. Registration images should be a xy single plane
        so many modalities (multi-channel, RGB) must "create" a single channel.
        Defaults: multi-channel images -> max intensity project image
        RGB -> greyscale then intensity inversion (black background, white foreground)
    channel_names: List[str]
        names for the channels to go into the OME-TIFF
    channel_colors: List[str]
        channels colors for OME-TIFF (not implemented)

    Returns
    -------
    reg_image: RegImage
        A RegImage subclass for the particular image loaded
    """
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
