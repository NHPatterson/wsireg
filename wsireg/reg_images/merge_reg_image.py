from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

import numpy as np

from wsireg.reg_images.loader import reg_image_loader


class MergeRegImage:
    def __init__(
        self,
        image_fp: List[Union[Path, str]],
        image_res: List[Union[int, float]],
        channel_names: Optional[List[List[str]]] = None,
        channel_colors: Optional[List[List[str]]] = None,
    ):
        if isinstance(image_fp, list) is False:
            raise ValueError(
                "MergeRegImage requires a list of images to merge"
            )

        if isinstance(image_res, list) is False:
            raise ValueError(
                "MergeRegImage requires a list of image resolutions for each image to merge"
            )

        if channel_names is None:
            channel_names = [None for _ in range(0, len(image_fp))]

        if channel_colors is None:
            channel_colors = [None for _ in range(0, len(image_fp))]

        images = []
        for im_idx, image_data in enumerate(
            zip(image_fp, image_res, channel_names, channel_colors)
        ):
            image, image_res, channel_names, channel_colors = image_data
            imdata = reg_image_loader(
                image,
                image_res,
                channel_names=channel_names,
                channel_colors=channel_colors,
            )
            if (
                imdata.channel_names is None
                or len(imdata.channel_names) != imdata.n_ch
            ):
                imdata._channel_names = [
                    f"C{idx}" for idx in range(0, imdata.n_ch)
                ]

            images.append(imdata)

        if all([im.im_dtype == images[0].im_dtype for im in images]) is False:
            warn(
                "MergeRegImage created with mixed data types, writing will cast "
                "to the largest data type"
            )

        if any([im.is_rgb for im in images]) is True:
            warn(
                "MergeRegImage does not support writing merged interleaved RGB "
                "Data will be written as multi-channel"
            )

        self.images = images
        self.image_fps = image_fp
        self.im_dtype = self.images[0].im_dtype

        self.is_rgb = False

        self.n_ch = np.sum([i.n_ch for i in self.images])
        self.channel_names = [i.channel_names for i in self.images]
        self.original_size_transform = None
