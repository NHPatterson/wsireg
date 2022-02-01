from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk
from tifffile import TiffWriter

from wsireg.reg_images import RegImage
from wsireg.reg_transform_seq import RegTransformSeq
from wsireg.utils.im_utils import (
    format_channel_names,
    get_pyramid_info,
    prepare_ome_xml_str,
)
from wsireg.utils.tile_image_transform import (
    add_tile_location_on_moving,
    compute_sub_res,
    determine_moving_tile_padding,
    generate_tile_coords,
    itk_transform_tiles,
    subres_zarr_to_tiles,
    tile_pad_output_size,
    zarr_to_tiles,
)


class OmeTiffWriter:
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
        reg_image: RegImage,
        reg_transform_seq: Optional[RegTransformSeq] = None,
    ):
        self.reg_image = reg_image
        self.reg_transform_seq = reg_transform_seq

    def _prepare_image_info(
        self,
        image_name,
        reg_transform_seq: Optional[RegTransformSeq] = None,
        write_pyramid=True,
        tile_size=512,
        compression="default",
        by_tile=False,
    ):

        if reg_transform_seq:
            self.x_size, self.y_size = reg_transform_seq.output_size
            self.x_spacing, self.y_spacing = reg_transform_seq.output_spacing
        else:
            self.y_size, self.x_size = (
                (self.reg_image.im_dims[0], self.reg_image.im_dims[1])
                if self.reg_image.is_rgb
                else (self.reg_image.im_dims[1], self.reg_image.im_dims[2])
            )
            self.y_spacing, self.x_spacing = None, None

        self.tile_size = tile_size
        # protect against too large tile size
        while (
            self.y_size / self.tile_size <= 1
            or self.x_size / self.tile_size <= 1
        ):
            self.tile_size = self.tile_size // 2

        if by_tile:
            self.y_size, self.x_size = tile_pad_output_size(
                self.y_size, self.x_size, tile_size=self.tile_size
            )

        self.pyr_levels, _ = get_pyramid_info(
            self.y_size, self.x_size, self.reg_image.n_ch, self.tile_size
        )
        self.n_pyr_levels = len(self.pyr_levels)

        if reg_transform_seq:
            self.PhysicalSizeY = self.y_spacing
            self.PhysicalSizeX = self.x_spacing
        else:
            self.PhysicalSizeY = self.reg_image.image_res
            self.PhysicalSizeX = self.reg_image.image_res

        channel_names = format_channel_names(
            self.reg_image.channel_names, self.reg_image.n_ch
        )

        self.omexml = prepare_ome_xml_str(
            self.y_size,
            self.x_size,
            self.reg_image.n_ch,
            self.reg_image.im_dtype,
            self.reg_image.is_rgb,
            PhysicalSizeX=self.PhysicalSizeX,
            PhysicalSizeY=self.PhysicalSizeY,
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
            Name=image_name,
            Channel=None if self.reg_image.is_rgb else {"Name": channel_names},
        )

        self.subifds = self.n_pyr_levels - 1 if write_pyramid is True else None

        if compression == "default":
            print("using default compression")
            self.compression = "jpeg" if self.reg_image.is_rgb else "deflate"
        else:
            self.compression = compression

    def write_image_by_plane(
        self,
        image_name,
        output_dir="",
        write_pyramid=True,
        tile_size=512,
        compression="default",
    ):

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")
        self._prepare_image_info(
            image_name,
            reg_transform_seq=self.reg_transform_seq,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
        )

        rgb_im_data = []

        print(f"saving to {output_file_name}")
        with TiffWriter(output_file_name, bigtiff=True) as tif:
            if self.reg_image.reader == "sitk":
                self.reg_image.read_full_image()
            for channel_idx in range(self.reg_image.n_ch):
                print(f"transforming : {channel_idx}")
                if self.reg_image.reader != "sitk":
                    image = self.reg_image.read_single_channel(channel_idx)
                    image = np.squeeze(image)
                    image = sitk.GetImageFromArray(image)
                    image.SetSpacing(
                        (self.reg_image.image_res, self.reg_image.image_res)
                    )

                if self.reg_transform_seq:
                    image = self.reg_transform_seq.resampler.Execute(image)
                    # image = transform_plane(
                    #     image, final_transform, composite_transform
                    # )
                    print(f"transformed : {channel_idx}")

                if self.reg_image.is_rgb:
                    rgb_im_data.append(image)
                else:
                    print("saving")
                    if isinstance(image, sitk.Image):
                        image = sitk.GetArrayFromImage(image)

                    options = dict(
                        tile=(self.tile_size, self.tile_size),
                        compression=self.compression,
                        photometric="rgb"
                        if self.reg_image.is_rgb
                        else "minisblack",
                        metadata=None,
                    )
                    # write OME-XML to the ImageDescription tag of the first page
                    description = self.omexml if channel_idx == 0 else None
                    # write channel data
                    print(
                        f" writing channel {channel_idx} - shape: {image.shape}"
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
                            print(
                                f"pyramid index {pyr_idx} : channel {channel_idx} shape: {image.shape}"
                            )

                            tif.write(image, **options, subfiletype=1)

            if self.reg_image.is_rgb:
                rgb_im_data = sitk.Compose(rgb_im_data)
                rgb_im_data = sitk.GetArrayFromImage(rgb_im_data)

                options = dict(
                    tile=(self.tile_size, self.tile_size),
                    compression=self.compression,
                    photometric="rgb",
                    metadata=None,
                )
                # write OME-XML to the ImageDescription tag of the first page
                description = self.omexml

                # write channel data
                tif.write(
                    rgb_im_data,
                    subifds=self.subifds,
                    description=description,
                    **options,
                )

                print(f"RGB shape: {rgb_im_data.shape}")
                if write_pyramid:
                    for pyr_idx in range(1, self.n_pyr_levels):
                        resize_shape = (
                            self.pyr_levels[pyr_idx][0],
                            self.pyr_levels[pyr_idx][1],
                        )
                        rgb_im_data = cv2.resize(
                            rgb_im_data,
                            resize_shape,
                            cv2.INTER_LINEAR,
                        )
                        tif.write(rgb_im_data, **options, subfiletype=1)
        return output_file_name

    def write_image_by_tile(
        self,
        image_name,
        output_dir="",
        write_pyramid=True,
        tile_size=512,
        compression="default",
        tile_padding=None,
        use_multiprocessing=False,
    ):

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")

        self._prepare_image_info(
            image_name,
            reg_transform_seq=self.reg_transform_seq,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
            by_tile=True,
        )

        # create per tile coordinate information for fixed
        # and moving
        tile_coordinate_data = generate_tile_coords(
            (self.x_size, self.y_size),
            (self.PhysicalSizeX, self.PhysicalSizeY),
            tile_size=self.tile_size,
        )
        # itk_transforms = [
        #     t.itk_transform for t in self.reg_transform_seq.reg_transforms
        # ]

        tile_coordinate_data = add_tile_location_on_moving(
            self.reg_transform_seq.reg_transforms_itk_order,
            tile_coordinate_data,
            self.reg_image.image_res,
        )

        if tile_padding is None:
            tile_padding = determine_moving_tile_padding(
                self.reg_transform_seq.reg_transforms_itk_order
            )

        print(f"saving to {output_file_name}")
        with TiffWriter(output_file_name, bigtiff=True) as tif:
            options = dict(
                tile=(self.tile_size, self.tile_size),
                compression=self.compression,
                photometric="rgb" if self.reg_image.is_rgb else "minisblack",
                metadata=None,
            )
            if self.reg_image.is_rgb:
                resampled_zarray = itk_transform_tiles(
                    self.reg_image,
                    self.y_size,
                    self.x_size,
                    tile_coordinate_data,
                    self.reg_transform_seq.composite_transform,
                    tile_size=self.tile_size,
                    tile_padding=tile_padding,
                    use_multiprocessing=use_multiprocessing,
                )
                print(
                    f"writing base layer RGB - shape: {resampled_zarray.shape}"
                )
                description = self.omexml
                tif.write(
                    zarr_to_tiles(
                        resampled_zarray,
                        tile_coordinate_data,
                        self.reg_image.is_rgb,
                    ),
                    subifds=self.subifds,
                    description=description,
                    shape=(self.y_size, self.x_size, 3),
                    dtype=resampled_zarray.dtype,
                    **options,
                )
                if write_pyramid:
                    for pyr_idx in range(1, self.n_pyr_levels):

                        resampled_zarray_subres, orig_shape = compute_sub_res(
                            self.reg_image,
                            resampled_zarray,
                            2**pyr_idx,
                            self.tile_size,
                            self.reg_image.is_rgb,
                        )
                        print(f"pyr {pyr_idx} : RGB-shape: {orig_shape}")

                        tif.write(
                            subres_zarr_to_tiles(
                                resampled_zarray_subres,
                                self.tile_size,
                                self.reg_image.is_rgb,
                            ),
                            shape=orig_shape,
                            dtype=resampled_zarray_subres.dtype,
                            **options,
                            subfiletype=1,
                        )
                try:
                    resampled_zarray.store.clear()
                except FileNotFoundError:
                    pass
            else:
                for channel_idx in range(self.reg_image.n_ch):
                    resampled_zarray = itk_transform_tiles(
                        self.reg_image,
                        self.y_size,
                        self.x_size,
                        tile_coordinate_data,
                        self.reg_transform_seq.composite_transform,
                        ch_idx=channel_idx,
                        tile_size=self.tile_size,
                        tile_padding=tile_padding,
                        use_multiprocessing=use_multiprocessing,
                    )

                    description = self.omexml if channel_idx == 0 else None
                    print(
                        f"writing channel {channel_idx} - shape: {resampled_zarray.shape}"
                    )
                    tif.write(
                        zarr_to_tiles(
                            resampled_zarray,
                            tile_coordinate_data,
                            self.reg_image.is_rgb,
                        ),
                        subifds=self.subifds,
                        description=description,
                        shape=(self.y_size, self.x_size),
                        dtype=resampled_zarray.dtype,
                        **options,
                    )
                    if write_pyramid:
                        for pyr_idx in range(1, self.n_pyr_levels):
                            (
                                resampled_zarray_subres,
                                orig_shape,
                            ) = compute_sub_res(
                                self.reg_image,
                                resampled_zarray,
                                2**pyr_idx,
                                self.tile_size,
                                self.reg_image.is_rgb,
                            )

                            tif.write(
                                subres_zarr_to_tiles(
                                    resampled_zarray_subres,
                                    self.tile_size,
                                    self.reg_image.is_rgb,
                                ),
                                shape=orig_shape,
                                dtype=resampled_zarray_subres.dtype,
                                **options,
                                subfiletype=1,
                            )
                    try:
                        resampled_zarray.store.clear()
                    except FileNotFoundError:
                        pass
        return output_file_name
