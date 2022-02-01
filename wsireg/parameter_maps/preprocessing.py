from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from pydantic import BaseModel, validator


class ImageType(str, Enum):
    """Set the photometric interpretation of the image
    * "FL": background is black (fluorescence)
    * "BF": Background is white (brightfield)
    """

    DARK = "FL"
    LIGHT = "BF"


class CoordinateFlip(str, Enum):
    """Coordinate flip options
    * "h" : horizontal flip
    * "v" : vertical flip
    """

    HORIZONTAL = "h"
    VERTIAL = "v"


class BoundingBox(NamedTuple):
    X: int
    Y: int
    WIDTH: int
    HEIGHT: int


def _transform_to_bbox(mask_bbox: Union[Tuple[int, int, int, int], List[int]]):
    return BoundingBox(*mask_bbox)


def _index_to_list(ch_indices: Union[int, List[int]]):
    if isinstance(ch_indices, int):
        ch_indices = [ch_indices]
    return ch_indices


def _transform_custom_proc(
    custom_procs: Union[List[Callable], Tuple[Callable, ...]]
):
    return {
        f"custom processing {str(idx+1).zfill(2)}": proc
        for idx, proc in enumerate(custom_procs)
    }


class ImagePreproParams(BaseModel):
    """Preprocessing parameter model

    Attributes
    ----------
    image_type: ImageType
        Whether image is dark or light background. Light background images are intensity inverted
        by default
    max_int_proj: bool
        Perform max intensity projection number of channels > 1.
    contrast_enhance: bool
        Enhance contrast of image
    invert_intensity: bool
        invert the intensity of an image
    rot_cc: int, float
        Rotate image counter-clockwise by degrees, can be positive or negative (cw rot)
    flip: CoordinateFlip, default: None
        flip coordinates, "v" = vertical flip, "h" = horizontal flip
    crop_to_mask_bbox: bool
        Convert a binary mask to a bounding box and crop to this area
    mask_bbox: tuple or list of 4 ints
        supply a pre-computed list of bbox info of form x,y,width,height
    """

    # intensity preprocessing
    image_type: ImageType = ImageType.DARK
    max_int_proj: bool = True
    ch_indices: Optional[List[int]] = None
    as_uint8: bool = True
    contrast_enhance: bool = False
    invert_intensity: bool = False
    custom_processing: Optional[Dict[str, Callable]] = None

    # spatial preprocessing
    rot_cc: Union[int, float] = 0
    flip: Optional[CoordinateFlip] = None
    crop_to_mask_bbox: bool = False
    mask_bbox: Optional[BoundingBox] = None
    downsampling: int = 1
    use_mask: bool = True

    @validator('mask_bbox', pre=True)
    def _make_bbox(cls, v):
        return _transform_to_bbox(v)

    @validator('ch_indices', pre=True)
    def _make_ch_list(cls, v):
        return _index_to_list(v)

    @validator('custom_processing', pre=True)
    def _check_custom_prepro(cls, v):
        if isinstance(v, (list, tuple)):
            return _transform_custom_proc(v)
        else:
            return v
