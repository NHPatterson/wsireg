from pathlib import Path
import numpy as np
from tifffile import TiffWriter
import cv2
import SimpleITK as sitk
from wsireg.utils.im_utils import (
    get_final_yx_from_tform,
    get_pyramid_info,
    format_channel_names,
    prepare_ome_xml_str,
    transform_plane,
)
from wsireg.utils.tile_image_transform import (
    add_tile_location_on_moving,
    determine_moving_tile_padding,
    generate_tile_coords,
    itk_transform_tiles,
    zarr_to_tiles,
    compute_sub_res,
    subres_zarr_to_tiles,
    tile_pad_output_size,
)


class OmeTiffWriter:
    def __init__(self, reg_image):
        self.reg_image = reg_image

    def prepare_image_info(
        self,
        image_name,
        final_transform=None,
        write_pyramid=True,
        tile_size=512,
        compression="default",
        by_tile=False,
    ):

        try:
            (
                self.y_size,
                self.x_size,
                self.y_spacing,
                self.x_spacing,
            ) = get_final_yx_from_tform(
                self.reg_image.images[0], final_transform[0]
            )
        except AttributeError:
            (
                self.y_size,
                self.x_size,
                self.y_spacing,
                self.x_spacing,
            ) = get_final_yx_from_tform(self.reg_image, final_transform)

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

        self.pyr_levels, self.pyr_shapes = get_pyramid_info(
            self.y_size, self.x_size, self.reg_image.n_ch, self.tile_size
        )
        self.n_pyr_levels = len(self.pyr_levels)

        if final_transform is not None:
            if isinstance(final_transform, list):
                if final_transform[0] is None:
                    self.PhysicalSizeY = self.reg_image.images[0].image_res
                    self.PhysicalSizeX = self.reg_image.images[0].image_res
                else:
                    self.PhysicalSizeY = self.y_spacing
                    self.PhysicalSizeX = self.x_spacing
            else:
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
        final_transform=None,
        composite_transform=None,
        write_pyramid=True,
        tile_size=512,
        compression="default",
    ):

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")
        self.prepare_image_info(
            image_name,
            final_transform=final_transform,
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

                if composite_transform is not None:
                    image = transform_plane(
                        image, final_transform, composite_transform
                    )
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
        final_transform=None,
        itk_transforms=None,
        composite_transform=None,
        write_pyramid=True,
        tile_size=512,
        compression="default",
        tile_padding=None,
        use_multiprocessing=False,
    ):

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")

        self.prepare_image_info(
            image_name,
            final_transform=final_transform,
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

        tile_coordinate_data = add_tile_location_on_moving(
            itk_transforms, tile_coordinate_data, self.reg_image.image_res
        )

        if tile_padding is None:
            tile_padding = determine_moving_tile_padding(itk_transforms)

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
                    composite_transform,
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
                            2 ** pyr_idx,
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
                        composite_transform,
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
                                2 ** pyr_idx,
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
