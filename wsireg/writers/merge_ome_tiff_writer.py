from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk
from tifffile import TiffWriter

from wsireg.reg_images import RegImage
from wsireg.reg_images.merge_reg_image import MergeRegImage
from wsireg.reg_transform_seq import RegTransformSeq
from wsireg.utils.im_utils import (
    SITK_TO_NP_DTYPE,
    format_channel_names,
    get_pyramid_info,
    prepare_ome_xml_str,
)


class MergeOmeTiffWriter:
    x_size: Optional[int] = None
    y_size: Optional[int] = None
    y_spacing: Optional[Union[int, float]] = None
    x_spacing: Optional[Union[int, float]] = None
    tile_size: int = 512
    pyr_levels: Optional[List[Tuple[int, int]]] = None
    n_pyr_levels: Optional[int] = None
    PhysicalSizeY: Optional[Union[int, float]] = None
    PhysicalSizeX: Optional[Union[int, float]] = None
    subifds: Optional[int] = None
    compression: str = "deflate"

    def __init__(
        self,
        reg_images: MergeRegImage,
        reg_seq_transforms: Optional[List[RegTransformSeq]] = None,
    ):
        self.reg_image = reg_images
        self.reg_seq_transforms = reg_seq_transforms

    def _length_checks(self, sub_image_names):

        if isinstance(sub_image_names, list) is False:
            if sub_image_names is None:
                sub_image_names = [
                    "" for i in range(len(self.reg_image.images))
                ]
            else:
                raise ValueError(
                    "MergeRegImage requires a list of image names for each image to merge"
                )

        if self.reg_seq_transforms is None:
            transformations = [None for i in range(len(self.reg_image.images))]
        else:
            transformations = self.reg_seq_transforms

        if len(transformations) != len(self.reg_image.images):
            raise ValueError(
                "MergeRegImage number of transforms does not match number of images"
            )

    def _create_channel_names(self, sub_image_names):
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

    def _transform_check(self):
        out_size = []
        out_spacing = []

        if not self.reg_seq_transforms:
            rts = [None for _ in range(len(self.reg_image.images))]
        else:
            rts = self.reg_seq_transforms
        for im, t in zip(self.reg_image.images, rts):
            if t:
                out_size.append(t.reg_transforms[-1].output_size)
                out_spacing.append(t.reg_transforms[-1].output_spacing)
            else:
                out_im_size = (
                    (im.im_dims[0], im.im_dims[1])
                    if im.is_rgb
                    else (im.im_dims[1], im.im_dims[2])
                )
                out_im_spacing = im.image_res

                out_size.append(out_im_size)
                out_spacing.append(out_im_spacing)

        if all(out_spacing) is False:
            raise ValueError(
                "MergeRegImage all transforms output spacings and untransformed image spacings must match"
            )

        if all(out_size) is False:
            raise ValueError(
                "MergeRegImage all transforms output sizes and untransformed image sizes must match"
            )

    def _prepare_image_info(
        self,
        reg_image: RegImage,
        image_name: str,
        im_dtype: np.dtype,
        reg_transform_seq: Optional[RegTransformSeq] = None,
        channel_names: Optional[List[str]] = None,
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str = "default",
    ):

        if reg_transform_seq:
            self.x_size, self.y_size = reg_transform_seq.output_size
            self.x_spacing, self.y_spacing = reg_transform_seq.output_spacing
        else:
            self.y_size, self.x_size = (
                (reg_image.im_dims[0], reg_image.im_dims[1])
                if reg_image.is_rgb
                else (reg_image.im_dims[1], reg_image.im_dims[2])
            )
            self.y_spacing, self.x_spacing = (
                reg_image.image_res,
                reg_image.image_res,
            )

        self.tile_size = tile_size
        # protect against too large tile size
        while (
            self.y_size / self.tile_size <= 1
            or self.x_size / self.tile_size <= 1
        ):
            self.tile_size = self.tile_size // 2

        self.pyr_levels, _ = get_pyramid_info(
            self.y_size, self.x_size, reg_image.n_ch, self.tile_size
        )
        self.n_pyr_levels = len(self.pyr_levels)

        if reg_transform_seq:
            self.PhysicalSizeY = self.y_spacing
            self.PhysicalSizeX = self.x_spacing
        else:
            self.PhysicalSizeY = reg_image.image_res
            self.PhysicalSizeX = reg_image.image_res

        channel_names = format_channel_names(
            self.reg_image.channel_names, self.reg_image.n_ch
        )

        self.omexml = prepare_ome_xml_str(
            self.y_size,
            self.x_size,
            len(channel_names),
            im_dtype,
            False,
            PhysicalSizeX=self.PhysicalSizeX,
            PhysicalSizeY=self.PhysicalSizeY,
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
            Name=image_name,
            Channel={"Name": channel_names},
        )

        self.subifds = self.n_pyr_levels - 1 if write_pyramid is True else None

        if compression == "default":
            print("using default compression")
            self.compression = "deflate"
        else:
            self.compression = compression

    def _get_merge_dtype(self):
        dtype_max_size = [
            np.iinfo(r.im_dtype).max for r in self.reg_image.images
        ]

        merge_dtype_np = self.reg_image.images[
            np.argmax(dtype_max_size)
        ].im_dtype
        for k, v in SITK_TO_NP_DTYPE.items():
            if k < 12:
                if v == merge_dtype_np:
                    merge_dtype_sitk = k
        return merge_dtype_sitk, merge_dtype_np

    def merge_write_image_by_plane(
        self,
        image_name,
        sub_image_names,
        output_dir="",
        write_pyramid=True,
        tile_size=512,
        compression="default",
    ):
        merge_dtype_sitk, merge_dtype_np = self._get_merge_dtype()

        self._length_checks(sub_image_names)
        self._create_channel_names(sub_image_names)
        self._transform_check()

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")

        self._prepare_image_info(
            self.reg_image.images[0],
            image_name,
            merge_dtype_np,
            reg_transform_seq=self.reg_seq_transforms[0],
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

                    if image.GetPixelIDValue() != merge_dtype_sitk:
                        image = sitk.Cast(image, merge_dtype_sitk)

                    if self.reg_seq_transforms[m_idx]:
                        image = self.reg_seq_transforms[
                            m_idx
                        ].resampler.Execute(image)

                    if isinstance(image, sitk.Image):
                        image = sitk.GetArrayFromImage(image)

                    options = dict(
                        tile=(tile_size, tile_size),
                        compression=self.compression,
                        photometric="minisblack",
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
                        f"channel index - {channel_idx} - shape: {image.shape}"
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
