from warnings import warn

import numpy as np

from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_images import RegImage
from wsireg.utils.im_utils import transform_to_ome_tiff_merge
from wsireg.utils.tform_utils import prepare_wsireg_transform_data


class MergeRegImage(RegImage):
    def __init__(
        self,
        image_fp,
        image_res,
        channel_names=None,
        channel_colors=None,
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
            channel_names = [None for i in range(0, len(image_fp))]

        if channel_colors is None:
            channel_colors = [None for i in range(0, len(image_fp))]

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
                imdata.channel_names = [
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

        # rgb merging not supported!
        self.is_rgb = False

        self.n_ch = np.sum([i.n_ch for i in self.images])
        self.channel_names = [i.channel_names for i in self.images]
        self.original_size_transform = None

    def transform_image(
        self,
        image_name,
        sub_image_names,
        transform_data,
        file_writer="ome.tiff",
        output_dir="",
        **transformation_opts,
    ):
        if isinstance(sub_image_names, list) is False:
            raise ValueError(
                "MergeRegImage requires a list of image names for each image to merge"
            )

        if isinstance(transform_data, list) is False:
            raise ValueError(
                "MergeRegImage requires a list of image transforms for each image to merge"
            )

        if len(transform_data) != len(self.image_fps):
            raise ValueError(
                "MergeRegImage number of transforms does not match number of images"
            )

        def get_transform_data(transform_data):
            if transform_data is not None:
                (
                    itk_composite,
                    itk_transforms,
                    final_transform,
                ) = prepare_wsireg_transform_data(transform_data)
            else:
                itk_composite, itk_transforms, final_transform = (
                    None,
                    None,
                    None,
                )

            return itk_composite, itk_transforms, final_transform

        def prepare_channel_names(sub_image_name, channel_names):
            return [f"{sub_image_name} - {c}" for c in channel_names]

        self.channel_names = [
            prepare_channel_names(im_name, cnames)
            for im_name, cnames in zip(sub_image_names, self.channel_names)
        ]
        self.channel_names = [
            item for sublist in self.channel_names for item in sublist
        ]

        merge_transform_data = [get_transform_data(t) for t in transform_data]
        composite_transforms = [
            merge_transform_data[t][0]
            for t in range(len(merge_transform_data))
        ]
        final_transforms = [
            merge_transform_data[t][2]
            for t in range(len(merge_transform_data))
        ]

        spacing_check = [
            t.output_spacing for t in final_transforms if t is not None
        ]
        size_check = [t.output_size for t in final_transforms if t is not None]
        no_transform_check = [
            idx for idx, t in enumerate(final_transforms) if t is None
        ]

        for no_tform in no_transform_check:
            size_check.extend(
                [np.asarray(self.images[no_tform].im_dims)[1:].tolist()]
            )
            spacing_check.extend(
                [
                    [
                        self.images[no_tform].image_res,
                        self.images[no_tform].image_res,
                    ]
                ]
            )

        if all(spacing_check) is False:
            raise ValueError(
                "MergeRegImage all transforms output spacings and untransformed image spacings must match"
            )

        if all(size_check) is False:
            raise ValueError(
                "MergeRegImage all transforms output sizes and untransformed image sizes must match"
            )

        if file_writer.lower() == "ome.tiff":
            im_fp = transform_to_ome_tiff_merge(
                self,
                image_name,
                output_dir,
                final_transforms,
                composite_transforms,
                **transformation_opts,
            )
        return im_fp
