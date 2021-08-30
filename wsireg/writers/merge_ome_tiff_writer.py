from pathlib import Path
import numpy as np
from tifffile import TiffWriter
import cv2
import SimpleITK as sitk
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_images.merge_reg_image import MergeRegImage
from wsireg.utils.im_utils import (
    transform_plane,
)
from wsireg.utils.tform_utils import (
    prepare_wsireg_transform_data,
)


class MergeOmeTiffWriter(OmeTiffWriter):
    def __init__(self, reg_image: MergeRegImage):
        super().__init__(reg_image)

    def merge_write_image_by_plane(
        self,
        image_name,
        sub_image_names,
        transformations=None,
        output_dir="",
        write_pyramid=True,
        tile_size=512,
        compression="default",
    ):

        if isinstance(sub_image_names, list) is False:
            if sub_image_names is None:
                sub_image_names = [
                    "" for i in range(len(self.reg_image.images))
                ]
            else:
                raise ValueError(
                    "MergeRegImage requires a list of image names for each image to merge"
                )

        if isinstance(transformations, list) is False:
            if transformations is None:
                transformations = [
                    None for i in range(len(self.reg_image.images))
                ]
            else:
                raise ValueError(
                    "MergeRegImage requires a list of image transforms for each image to merge"
                )

        if len(transformations) != len(self.reg_image.images):
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

        self.reg_image.channel_names = [
            prepare_channel_names(im_name, cnames)
            for im_name, cnames in zip(
                sub_image_names, self.reg_image.channel_names
            )
        ]
        self.reg_image.channel_names = [
            item
            for sublist in self.reg_image.channel_names
            for item in sublist
        ]

        merge_transform_data = [get_transform_data(t) for t in transformations]
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
                [
                    np.asarray(self.reg_image.images[no_tform].im_dims)[
                        1:
                    ].tolist()
                ]
            )
            spacing_check.extend(
                [
                    [
                        self.reg_image.images[no_tform].image_res,
                        self.reg_image.images[no_tform].image_res,
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

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")

        self.prepare_image_info(
            image_name,
            final_transform=final_transforms,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
        )

        print(f"saving to {output_file_name}")
        with TiffWriter(output_file_name, bigtiff=True) as tif:
            for m_idx, merge_image in enumerate(self.reg_image.images):
                if self.reg_image.images[m_idx].reader == "sitk":
                    full_image = sitk.ReadImage(
                        self.reg_image.images[m_idx].image_filepath
                    )
                merge_n_ch = merge_image.n_ch
                for channel_idx in range(merge_n_ch):
                    if self.reg_image.images[m_idx].reader != "sitk":
                        image = self.reg_image.images[
                            m_idx
                        ].read_single_channel(channel_idx)
                        image = np.squeeze(image)
                        image = sitk.GetImageFromArray(image)
                        image.SetSpacing(
                            (
                                self.reg_image.images[m_idx].image_res,
                                self.reg_image.images[m_idx].image_res,
                            )
                        )
                    else:
                        if len(full_image.GetSize()) > 2:
                            image = full_image[:, :, channel_idx]
                        else:
                            image = full_image

                    if composite_transforms[m_idx] is not None:
                        image = transform_plane(
                            image,
                            final_transforms[m_idx],
                            composite_transforms[m_idx],
                        )

                    if isinstance(image, sitk.Image):
                        image = sitk.GetArrayFromImage(image)

                    options = dict(
                        tile=(tile_size, tile_size),
                        compression=self.compression,
                        photometric="rgb"
                        if merge_image.is_rgb
                        else "minisblack",
                        metadata=None,
                    )
                    # write OME-XML to the ImageDescription tag of the first page
                    description = (
                        self.omexml
                        if channel_idx == 0 and m_idx == 0
                        else None
                    )
                    # write channel data
                    print(
                        f" writing subimage index {m_idx} : {sub_image_names[m_idx]} - "
                        f"channel {channel_idx} - shape: {image.shape}"
                    )
                    tif.write(
                        image,
                        subifds=self.subifds,
                        description=description,
                        **options,
                    )

                    if write_pyramid:
                        for pyr_idx in range(1, self.n_pyr_levels):
                            resize_shape = (
                                self.pyr_levels[pyr_idx][0],
                                self.pyr_levels[pyr_idx][1],
                            )
                            image = cv2.resize(
                                image,
                                resize_shape,
                                cv2.INTER_LINEAR,
                            )

                            tif.write(image, **options, subfiletype=1)

            return output_file_name
