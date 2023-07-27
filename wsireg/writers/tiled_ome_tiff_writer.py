import multiprocessing
import random
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import SimpleITK as sitk
import zarr
from tifffile import TiffWriter
from tiler import Tiler
from tqdm import tqdm
from wsireg.reg_images.reg_image import RegImage
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.utils.im_utils import (
    format_channel_names,
    get_pyramid_info,
    prepare_ome_xml_str,
)
from wsireg.utils.tform_utils import ELX_TO_ITK_INTERPOLATORS


class OmeTiffTiledWriter:
    """
    Class for transforming, then writing whole slide images tile-by-tile, allowing
    memory-efficient transformation of images. The image
    to be transformed has to have a dask representation that is itself tiled because
    the writer finds the position of each write tile in fixed space in moving space
    and reads only the necessary portion of the image to
    perform the transformation.

    Tiles are stored in a temporary `zarr` store that is deleted after writing to OME-TIFF.

    Uses the Tiler library to manage "virtual tiling" for transformation.

    Parameters
    ----------
    reg_image: RegImage
        wsireg RegImage that has a dask store that is chunked in XY (typical of WSIs)
        .czi reader does not work!

    reg_transform_seq: RegTransformSeq
        wsireg registration transform sequence to be applied to the image

    tile_size: int
        Tile size of the output image

    zarr_tile_size: int
        Tile used in the zarr intermediate

    moving_tile_padding: int
        How much additional padded to pull from moving for each transformed tile
        This ensures that the interpolation is correctly performed during resampling
        Non-rigid transforms may need more spacing

    Attributes
    ----------
    reg_image: RegImage
        RegImage to be transformed
    reg_transform_seq: RegTransformSeq
        RegTransformSeq to be used in transformation
    tile_shape: tuple of ints
        Shape of OME-TIFF tiles going to disk
    zarr_tile_shape: tuple of ints
        Shape of zarr tiles going to disk temporarily
    moving_tile_padding: int
        Tile padding use at read in for interpolation
    """

    def __init__(
        self,
        reg_image: RegImage,
        reg_transform_seq: RegTransformSeq,
        tile_size: int = 512,
        zarr_tile_size: int = 2048,
        moving_tile_padding: int = 128,
    ):
        self._fixed_tile_positions: List[Tuple[int, int, int, int]] = []
        self._fixed_tile_positions_phys: List[
            Tuple[float, float, float, float]
        ] = []
        self._moving_tile_positions: List[Tuple[int, int, int, int]] = []
        self._moving_tile_positions_phys: List[
            Tuple[float, float, float, float]
        ] = []
        self._tiler: Optional[Tiler] = None

        self.reg_image: RegImage = reg_image
        self.reg_transform_seq: RegTransformSeq = reg_transform_seq
        self.tile_shape = (tile_size, tile_size)
        self.zarr_tile_shape = (zarr_tile_size, zarr_tile_size)
        self._check_dask_array_chunk_sizes(self.reg_image.dask_image)
        self.moving_tile_padding = moving_tile_padding
        self._build_transformation_tiles()

    @property
    def fixed_tile_positions(self) -> List[Tuple[int, int, int, int]]:
        """List of tile positions on the fixed image in pixels
        first np.ndarray is top-left x,y coordinate
        second np.ndarray is bottom-right x,y coordinate"""
        return self._fixed_tile_positions

    @property
    def fixed_tile_positions_phys(
        self,
    ) -> List[Tuple[float, float, float, float]]:
        """List of tile positions on the fixed image in physical coordinate space
        first np.ndarray is top-left x,y coordinate
        second np.ndarray is bottom-right x,y coordinate"""
        return self._fixed_tile_positions_phys

    @property
    def moving_tile_positions(self) -> List[Tuple[int, int, int, int]]:
        """Transformed coordinates of fixed tile positions to moving, pixels
        first np.ndarray is top-left x,y coordinate
        second np.ndarray is bottom-right x,y coordinate"""
        return self._moving_tile_positions

    @property
    def moving_tile_positions_phys(
        self,
    ) -> List[Tuple[float, float, float, float]]:
        """Transformed coordinates of fixed tile positions to moving, physical
        first np.ndarray is top-left x,y coordinate
        second np.ndarray is bottom-right x,y coordinate"""
        return self._moving_tile_positions_phys

    @property
    def tiler(self) -> Tiler:
        """Tiler instance to manage fixed output tiling from image shape."""
        return self._tiler

    def _create_tiler(self):
        """Create the Tiler instance."""
        self._tiler = Tiler(
            self.reg_transform_seq.output_size,
            self.zarr_tile_shape,
            overlap=0,
            mode="irregular",
        )

    def _check_dask_array_chunk_sizes(self, dask_image: da.Array) -> None:
        """Check if dask image has an acceptable chunk-size for tiled writing."""
        yx_chunks = (
            dask_image.chunksize[:2]
            if self.reg_image.is_rgb
            else dask_image.chunksize[1:]
        )

        if np.any(np.asarray(yx_chunks) > np.asarray(self.zarr_tile_shape)):
            raise ValueError(
                f"Dask image chunksize for image {str(self.reg_image.path)} "
                "is too large for tiled writing and effectively memory use is not "
                "compared to plane-by-plane writing."
            )

        return

    def _build_transformation_tiles(self):
        """Method to reinitialize tiler if there are changes."""
        self._create_tiler()
        self._get_fixed_tile_positions()
        self._get_fixed_tile_positions_phys()
        self._get_moving_tile_positions()

    def _get_and_clip_fixed_tile(
        self, tile_idx: int, output_size: Tuple[int, int], order=[1, 0]
    ):
        """Method to ensure tiles do not go beyond the output shape
        of the fixed target image"""
        tile_pos = self.tiler.get_tile_bbox(tile_idx)
        if tile_pos[1][0] > output_size[order[0]]:
            tile_pos[1][0] = output_size[order[0]]
        if tile_pos[1][1] > output_size[order[1]]:
            tile_pos[1][1] = output_size[order[1]]

        return tile_pos

    def _get_fixed_tile_positions(self):
        """Find the tile positions on the fixed image."""
        self._fixed_tile_positions = [
            self._get_and_clip_fixed_tile(
                i, self.reg_transform_seq.output_size, order=[0, 1]
            )
            for i in range(self.tiler.n_tiles)
        ]

    def _get_fixed_tile_positions_phys(self):
        """Fixed tile pixel indices to physical coordinates
        used in ITK transforms."""
        self._fixed_tile_positions_phys = [
            (
                f[0] * self.reg_transform_seq.output_spacing,
                f[1] * self.reg_transform_seq.output_spacing,
            )
            for f in self._fixed_tile_positions
        ]

    def _get_moving_tile_positions(self):
        """Method to transform tile positions in fixed
        to moving so that each write tile in fixed has a corresponding
        read region in moving."""
        for fixed_tile_pos in self._fixed_tile_positions_phys:
            corners_phys = []
            corners_px = []

            for idx, corner in enumerate(fixed_tile_pos):
                if idx == 0:
                    corner -= self.moving_tile_padding
                if idx == 1:
                    corner += self.moving_tile_padding
                for idx, t in enumerate(
                    self.reg_transform_seq.reg_transforms[::-1]
                ):
                    if idx == 0:
                        t_pt = t.itk_transform.TransformPoint(corner.tolist())
                    else:
                        t_pt = t.itk_transform.TransformPoint(t_pt)

                t_pt = np.array(t_pt)
                t_pt_px = t_pt / self.reg_image.image_res
                corners_phys.append(t_pt)
                corners_px.append(t_pt_px)

            self._moving_tile_positions_phys.append(tuple(corners_phys))
            self._moving_tile_positions.append(tuple(corners_px))

    def set_output_spacing(
        self, output_spacing: Tuple[Union[int, float], Union[int, float]]
    ) -> None:
        """
        Sets the output spacing of the resampled image and will change
        output shape accordingly
        Parameters
        ----------
        output_spacing: Tuple[Union[int,float], Union[int,float]]
            Spacing of grid for resampling. Will default to target image spacing
        """
        self.reg_transform_seq.set_output_spacing(output_spacing)
        self._build_transformation_tiles()

    def set_tile_size(self, tile_size: int) -> None:
        """
        Set the internal tile size of the OME-TIFF to be written.
        Parameters
        ----------
        tile_size: int
            tile size in pixels in x and y for the OME-TIFF

        """
        self.tile_shape = (tile_size, tile_size)

    def set_zarr_tile_size(self, tile_size: int) -> None:
        """
        Set the tile size for the zarr intermediate.
        Parameters
        ----------
        tile_size: int
            tile size in pixels in x and y for the temporary zarr store
        """
        self.zarr_tile_shape = (tile_size, tile_size)
        self._build_transformation_tiles()

    def _create_tile_resampler(
        self, tile_origin: Tuple[float, float]
    ) -> sitk.ResampleImageFilter:
        """
        Build each tile's resampler.
        Parameters
        ----------
        tile_origin: Tuple[float, float]
            Position of the tile in physical coordinates

        Returns
        -------
        resampler: sitk.ResampleImageFilter
            resampler for an individual fixed tile
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(tile_origin)
        resampler.SetOutputDirection(
            self.reg_transform_seq.reg_transforms[-1].output_direction
        )
        resampler.SetSize(self.zarr_tile_shape)
        resampler.SetOutputSpacing(self.reg_transform_seq.output_spacing)

        interpolator = ELX_TO_ITK_INTERPOLATORS.get(
            self.reg_transform_seq.reg_transforms[-1].resample_interpolator
        )
        resampler.SetInterpolator(interpolator)
        resampler.SetTransform(self.reg_transform_seq.composite_transform)

        return resampler

    def write_tiles_to_zarr_store(
        self,
        temp_zarr_store: zarr.TempStore,
        max_workers: Optional[int] = None,
    ):
        """
        Write tiles to a temporary zarr store.

        Parameters
        ----------
        temp_zarr_store: zarr.TempStore
            Temporary store where the dataset will go.

        Returns
        -------
        resample_zarray: zarr.Array
            zarr store contained transformed images
        """
        zgrp = zarr.open(temp_zarr_store)
        if self.reg_image.is_rgb:
            resample_zarray = zgrp.create_dataset(
                random_str(),
                shape=(
                    self.reg_transform_seq.output_size[1],
                    self.reg_transform_seq.output_size[0],
                    self.reg_image.shape[-1],
                ),
                chunks=self.tile_shape,
                dtype=self.reg_image.im_dtype,
            )
        else:
            resample_zarray = zgrp.create_dataset(
                random_str(),
                shape=(
                    self.reg_image.n_ch,
                    self.reg_transform_seq.output_size[1],
                    self.reg_transform_seq.output_size[0],
                ),
                chunks=(1,) + self.tile_shape,
                dtype=self.reg_image.im_dtype,
            )

        self._transform_write_tile_set(
            resample_zarray, max_workers=max_workers
        )

        return resample_zarray

    def _transform_write_tile(self, data):
        """Worker function to transform and place tile in zarr store."""
        (
            resample_zarray,
            ch_idx,
            fixed_tile_position,
            fixed_tile_origin,
            moving_tile_corners,
        ) = data

        tile_resampler = self._create_tile_resampler(fixed_tile_origin)

        x_size, y_size = self._get_image_size()

        x_max, x_min, y_max, y_min = self._get_moving_tile_slice(
            moving_tile_corners, x_size, y_size
        )

        tile_resampled = self._resample_tile(
            ch_idx, tile_resampler, x_max, x_min, y_max, y_min
        )

        if tile_resampled:
            (
                x_max_fixed,
                x_min_fixed,
                y_max_fixed,
                y_min_fixed,
            ) = self._get_fixed_slice(fixed_tile_position)

            x_max, y_max = self._correct_end_moving_slices(
                x_max_fixed,
                x_min_fixed,
                y_max_fixed,
                y_min_fixed,
            )

            if self.reg_image.is_rgb:
                resample_zarray[
                    y_min_fixed:y_max_fixed, x_min_fixed:x_max_fixed, :
                ] = sitk.GetArrayFromImage(tile_resampled)[:y_max, :x_max, :]
            else:
                resample_zarray[
                    ch_idx, y_min_fixed:y_max_fixed, x_min_fixed:x_max_fixed
                ] = sitk.GetArrayFromImage(tile_resampled)[:y_max, :x_max]

    def _get_image_size(self) -> Tuple[int, int]:
        """Get moving image size for tile dilineation"""
        x_size = (
            self.reg_image.shape[1]
            if self.reg_image.is_rgb
            else self.reg_image.shape[2]
        )
        y_size = (
            self.reg_image.shape[0]
            if self.reg_image.is_rgb
            else self.reg_image.shape[1]
        )
        return x_size, y_size

    def _resample_tile(
        self,
        ch_idx: int,
        tile_resampler: sitk.ResampleImageFilter,
        x_max: int,
        x_min: int,
        y_max: int,
        y_min: int,
    ) -> Optional[sitk.Image]:
        """Resample tile or don't if it is outside of the moving
        image space."""

        if x_min == 0 and x_max == 0:
            return

        if y_min == 0 and y_max == 0:
            return

        if self.reg_image.is_rgb:
            image = self.reg_image.dask_image[y_min:y_max, x_min:x_max, :]
            image = sitk.GetImageFromArray(image, isVector=True)
        elif self.reg_image.n_ch == 1:
            image = da.squeeze(self.reg_image.dask_image)[
                y_min:y_max, x_min:x_max
            ]
            image = sitk.GetImageFromArray(image, isVector=False)
        else:
            image = self.reg_image.dask_image[ch_idx, y_min:y_max, x_min:x_max]
            image = sitk.GetImageFromArray(image, isVector=False)
        image.SetSpacing((self.reg_image.image_res, self.reg_image.image_res))
        image.SetOrigin(
            image.TransformIndexToPhysicalPoint([int(x_min), int(y_min)])
        )

        tile_resampled = tile_resampler.Execute(image)

        return tile_resampled

    def _correct_end_moving_slices(
        self,
        x_max_fixed: int,
        x_min_fixed: int,
        y_max_fixed: int,
        y_min_fixed: int,
    ) -> Tuple[int, int]:
        """Correct tiles that extend past the size of the fixed coordinate space."""
        # correct for end tiles
        y_max = y_max_fixed - y_min_fixed
        x_max = x_max_fixed - x_min_fixed
        return x_max, y_max

    def _get_fixed_slice(
        self, fixed_tile_position: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[int, int, int, int]:
        """Get tile slice in fixed tile pixels."""
        y_min_fixed = fixed_tile_position[0][1]
        x_min_fixed = fixed_tile_position[0][0]
        y_max_fixed = fixed_tile_position[1][1]
        x_max_fixed = fixed_tile_position[1][0]
        return x_max_fixed, x_min_fixed, y_max_fixed, y_min_fixed

    def _get_moving_tile_slice(
        self,
        moving_tile_corners: Tuple[np.ndarray, np.ndarray],
        x_size: int,
        y_size: int,
    ) -> Tuple[int, int, int, int]:
        """Get tile slice in moving tile pixels."""
        x_min = (
            moving_tile_corners[0][0] if moving_tile_corners[0][0] >= 0 else 0
        )
        x_min = x_min if x_min <= x_size else x_size
        x_min = np.ceil(x_min).astype(int)
        x_max = (
            moving_tile_corners[1][0] if moving_tile_corners[1][0] >= 0 else 0
        )
        x_max = x_max if x_max <= x_size else x_size
        x_max = np.ceil(x_max).astype(int)
        y_min = (
            moving_tile_corners[0][1] if moving_tile_corners[0][1] >= 0 else 0
        )
        y_min = y_min if y_min <= y_size else y_size
        y_min = np.ceil(y_min).astype(int)
        y_max = (
            moving_tile_corners[1][1] if moving_tile_corners[1][1] >= 0 else 0
        )
        y_max = y_max if y_max <= y_size else y_size
        y_max = np.ceil(y_max).astype(int)
        # catch changing positions of x and y when there are coordinate flips
        if y_min > y_max:
            y_max_temp = y_min
            y_min_temp = y_max
            y_max = y_max_temp
            y_min = y_min_temp
        if x_min > x_max:
            x_max_temp = x_min
            x_min_temp = x_max
            x_max = x_max_temp
            x_min = x_min_temp
        return x_max, x_min, y_max, y_min

    def _transform_write_tile_set(
        self, resample_zarray: zarr.Array, max_workers: Optional[int] = None
    ):
        """Function to loop over all channels and tile positions
        and write to zarr"""
        if max_workers == 1:
            use_multiprocessing = False
        else:
            use_multiprocessing = True
            if not max_workers:
                max_workers = multiprocessing.cpu_count()

        n_ch = 1 if self.reg_image.is_rgb else self.reg_image.n_ch
        all_tile_args = []
        for ch_idx in range(n_ch):
            for ft_pos, mt_pos in tqdm(
                zip(
                    self._fixed_tile_positions,
                    self._moving_tile_positions,
                ),
                total=len(self._fixed_tile_positions),
                desc="Writing zarr tiles",
                unit=" tile",
                disable=True if use_multiprocessing else False,
            ):
                tile_origin = (
                    ft_pos[0] * self.reg_transform_seq.output_spacing[0]
                )
                tile_args = (
                    resample_zarray,
                    ch_idx,
                    ft_pos,
                    tuple(tile_origin.astype(float)),
                    mt_pos,
                )
                all_tile_args.append(tile_args)
                if not use_multiprocessing:
                    self._transform_write_tile(tile_args)

        if use_multiprocessing:
            with ThreadPoolExecutor(max_workers) as executor:
                _ = list(
                    tqdm(
                        executor.map(
                            self._transform_write_tile, all_tile_args
                        ),
                        total=len(all_tile_args),
                        desc="Writing zarr tiles",
                        unit=" tile",
                    )
                )

    def _prepare_image_info(
        self,
        image_name,
        write_pyramid=True,
    ):
        """Prepare info for pyramidalization and create OME-TIFF."""

        x_size, y_size = self.reg_transform_seq.output_size
        x_spacing, y_spacing = self.reg_transform_seq.output_spacing

        out_tile_shape = self.tile_shape
        # protect against too large tile size
        while (
            y_size / out_tile_shape[0] <= 1 or x_size / out_tile_shape[0] <= 1
        ):
            out_tile_shape = (out_tile_shape[0] // 2, out_tile_shape[1] // 2)

        pyr_levels, _ = get_pyramid_info(
            y_size, x_size, self.reg_image.n_ch, self.tile_shape[0]
        )

        n_pyr_levels = len(pyr_levels)

        PhysicalSizeY = y_spacing
        PhysicalSizeX = x_spacing

        channel_names = format_channel_names(
            self.reg_image.channel_names, self.reg_image.n_ch
        )

        omexml = prepare_ome_xml_str(
            y_size,
            x_size,
            self.reg_image.n_ch,
            self.reg_image.im_dtype,
            self.reg_image.is_rgb,
            PhysicalSizeX=PhysicalSizeX,
            PhysicalSizeY=PhysicalSizeY,
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
            Name=image_name,
            Channel=None if self.reg_image.is_rgb else {"Name": channel_names},
        )

        subifds = n_pyr_levels - 1 if write_pyramid is True else None

        return n_pyr_levels, subifds, out_tile_shape, omexml

    def _transformed_tile_generator(self, d_array: da.Array, ch_idx: int):
        """Create generator of tifffile tiles for OME-TIFF."""
        out_shape = (
            d_array.shape[:2] if self.reg_image.is_rgb else d_array.shape[1:]
        )
        for y in range(0, out_shape[0], self.tile_shape[0]):
            for x in range(0, out_shape[1], self.tile_shape[1]):
                if self.reg_image.is_rgb:
                    yield d_array[
                        y : y + self.tile_shape[0],
                        x : x + self.tile_shape[1],
                        :,
                    ].compute()
                else:
                    yield d_array[
                        ch_idx,
                        y : y + self.tile_shape[0],
                        x : x + self.tile_shape[1],
                    ].compute()

    def write_image_by_tile(
        self,
        image_name: str,
        output_dir: Union[Path, str] = "",
        write_pyramid: bool = True,
        compression: Optional[str] = "default",
        zarr_temp_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Write images to OME-TIFF from temp zarr store with data.

        Parameters
        ----------
        image_name: str
            file path stem of the image to be written
        output_dir: Union[str,Path]
            directory where image is to be written
        write_pyramid: bool
            whether to write a pyramid or single layer
        compression: str
            Use compression. "default" will be lossless "deflate" for non-rgb images
            and "jpeg" for RGB images
        zarr_temp_dir: Path or str
            Directory to store the temporary zarr data
            (mostly used for debugging)

        Returns
        -------
        output_file_name: Path
            Path to written image file
        """
        zstr = zarr.TempStore(dir=zarr_temp_dir)
        try:
            resample_zarray = self.write_tiles_to_zarr_store(zstr)
            output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")

            if compression == "default":
                print("using default compression")
                compression = "jpeg" if self.reg_image.is_rgb else "deflate"
            else:
                compression = compression

            (
                n_pyr_levels,
                subifds,
                out_tile_shape,
                omexml,
            ) = self._prepare_image_info(
                image_name, write_pyramid=write_pyramid
            )

            print(f"saving to {output_file_name}")

            dask_image = da.from_zarr(resample_zarray)
            options = dict(
                tile=self.tile_shape,
                compression=compression,
                photometric="rgb" if self.reg_image.is_rgb else "minisblack",
                metadata=None,
            )
            with TiffWriter(output_file_name, bigtiff=True) as tif:
                if self.reg_image.is_rgb:
                    print(
                        f"writing base layer RGB - shape: {dask_image.shape}"
                    )
                    # tile_iterator_strides = self._get_tile_iterator_strides(dask_image)
                    tile_iterator = self._transformed_tile_generator(
                        dask_image, 0
                    )
                    tif.write(
                        tile_iterator,
                        subifds=subifds,
                        description=omexml,
                        shape=dask_image.shape,
                        dtype=dask_image.dtype,
                        **options,
                    )

                    if write_pyramid:
                        for pyr_idx in range(1, n_pyr_levels):
                            sub_res = compute_sub_res(
                                dask_image,
                                pyr_idx,
                                self.tile_shape[0],
                                self.reg_image.is_rgb,
                                self.reg_image.im_dtype,
                            )
                            print(
                                f"pyr {pyr_idx} : RGB-shape: {sub_res.shape}"
                            )

                            # tile_strides = self._get_tile_iterator_strides(sub_res)
                            sub_res_tile_iterator = (
                                self._transformed_tile_generator(sub_res, 0)
                            )
                            tif.write(
                                sub_res_tile_iterator,
                                shape=sub_res.shape,
                                dtype=self.reg_image.im_dtype,
                                **options,
                                subfiletype=1,
                            )
                else:
                    for channel_idx in range(self.reg_image.n_ch):
                        description = omexml if channel_idx == 0 else None
                        print(
                            f"writing channel {channel_idx} - shape: {dask_image.shape[1:]}"
                        )
                        tile_iterator = self._transformed_tile_generator(
                            dask_image, channel_idx
                        )

                        tif.write(
                            tile_iterator,
                            subifds=subifds,
                            description=description,
                            shape=dask_image.shape[1:],
                            dtype=dask_image.dtype,
                            **options,
                        )
                        if write_pyramid:
                            for pyr_idx in range(1, n_pyr_levels):
                                sub_res = compute_sub_res(
                                    dask_image,
                                    pyr_idx,
                                    self.tile_shape[0],
                                    self.reg_image.is_rgb,
                                    self.reg_image.im_dtype,
                                )

                                sub_res_tile_iterator = (
                                    self._transformed_tile_generator(
                                        sub_res, channel_idx
                                    )
                                )

                                tif.write(
                                    sub_res_tile_iterator,
                                    shape=sub_res.shape[1:],
                                    dtype=dask_image.dtype,
                                    **options,
                                    subfiletype=1,
                                )
            try:
                resample_zarray.store.clear()
            except FileNotFoundError:
                pass
            return output_file_name

        # bare except to always clear temporary storage on failure
        except Exception as e:
            print(e)
            try:
                resample_zarray.store.clear()
            except FileNotFoundError:
                pass


def compute_sub_res(
    zarray: da.Array,
    pyr_level: int,
    tile_size: int,
    is_rgb: bool,
    im_dtype: np.dtype,
) -> da.Array:
    """
    Compute factor-of-2 sub-resolutions from dask array for pyramidalization using dask.

    Parameters
    ----------
    zarray: da.Array
        Dask array to be downsampled
    pyr_level: int
        level of the pyramid. 0 = base, 1 = 2x downsampled, 2=4x downsampled...
    tile_size: int
        Size of tiles in dask array after downsampling
    is_rgb: bool
        whether dask array is RGB interleaved
    im_dtype: np.dtype
        dtype of the output da.Array

    Returns
    -------
    resampled_zarray_subres: da.Array
        Dask array (unprocessed) to be written
    """
    if is_rgb:
        resampling_axis = {0: 2**pyr_level, 1: 2**pyr_level, 2: 1}
        tiling = (tile_size, tile_size, 3)
    else:
        resampling_axis = {0: 1, 1: 2**pyr_level, 2: 2**pyr_level}
        tiling = (1, tile_size, tile_size)

    resampled_zarray_subres = da.coarsen(
        np.mean,
        zarray,
        resampling_axis,
        trim_excess=True,
    )
    resampled_zarray_subres = resampled_zarray_subres.astype(im_dtype)
    resampled_zarray_subres = resampled_zarray_subres.rechunk(tiling)

    return resampled_zarray_subres


def random_str() -> str:
    """Get a random string to store the zarr array"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(10))
