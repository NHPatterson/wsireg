from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk
from tifffile import TiffWriter

from wsireg.reg_images.reg_image import RegImage
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.utils.im_utils import (
    format_channel_names,
    get_pyramid_info,
    prepare_ome_xml_str,
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
        """
        Class for managing writing images to OME-TIFF.

        Parameters
        ----------
        reg_image: RegImage
            RegImage to be transformed
        reg_transform_seq: RegTransformSeq or None
            Registration transformation sequence from wsireg to transform image

        Attibutes
        ---------
        x_size: int
            Size of the output image after transformation in x
        y_size: int
            Size of the output image after transformation in y
        y_spacing: float
            Pixel spacing in microns after transformation in y
        x_spacing: float
            Pixel spacing in microns after transformation in x
        tile_size: int
            Size of tiles to be written
        pyr_levels: list of tuples of int:
            Size of downsampled images in pyramid
        n_pyr_levels: int
            Number of downsamples in pyramid
        PhysicalSizeY: float
            physical size of image in micron for OME-TIFF in Y
        PhysicalSizeX: float
            physical size of image in micron for OME-TIFF in X
        subifds: int
            Number of sub-resolutions for pyramidal OME-TIFF
        compression: str
            tifffile string to pass to compression argument, defaults to "deflate" for minisblack
            and "jpeg" for RGB type images

        """
        self.reg_image = reg_image
        self.reg_transform_seq = reg_transform_seq

    def _prepare_image_info(
        self,
        image_name: str,
        reg_transform_seq: Optional[RegTransformSeq] = None,
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: Optional[str] = "default",
    ) -> None:
        """Get image info and OME-XML"""

        if reg_transform_seq:
            self.x_size, self.y_size = reg_transform_seq.output_size
            self.x_spacing, self.y_spacing = reg_transform_seq.output_spacing
        else:
            self.y_size, self.x_size = (
                (self.reg_image.shape[0], self.reg_image.shape[1])
                if self.reg_image.is_rgb
                else (self.reg_image.shape[1], self.reg_image.shape[2])
            )
            self.y_spacing, self.x_spacing = None, None

        self.tile_size = tile_size
        # protect against too large tile size
        while (
            self.y_size / self.tile_size <= 1
            or self.x_size / self.tile_size <= 1
        ):
            self.tile_size = self.tile_size // 2

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
        image_name: str,
        output_dir: Union[Path, str] = "",
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: Optional[str] = "default",
        interpolation: int = cv2.INTER_LINEAR,
        blur_size: int = 0,
        subsample: bool = False,
    ) -> str:
        """
        Write OME-TIFF image plane-by-plane to disk. WsiReg compatible RegImages all
        have methods to read an image channel-by-channel, thus each channel is read, transformed, and written to
        reduce memory during write.
        RGB images may run large memory footprints as they are interleaved before write, for RGB images,
        using the `OmeTiledTiffWriter` is recommended.

        Parameters
        ----------
        image_name: str
            Name to be written WITHOUT extension
            for example if image_name = "cool_image" the file
            would be "cool_image.ome.tiff"
        output_dir: Path or str
            Directory where the image will be saved
        write_pyramid: bool
            Whether to write the OME-TIFF with sub-resolutions or not
        tile_size: int
            What size to write OME-TIFF tiles to disk
        compression: str
            tifffile string to pass to compression argument, defaults to "deflate" for minisblack
            and "jpeg" for RGB type images

        Returns
        -------
        output_file_name: str
            File path to the written OME-TIFF

        """

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
                self.reg_image._read_full_image()

            for channel_idx in range(self.reg_image.n_ch):
                print(f"transforming : {channel_idx}")
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
                            if subsample:
                                image = image[::2, ::2]
                            else:
                                if blur_size > 0:
                                    image = cv2.blur(
                                        image, (blur_size, blur_size)
                                    )
                                # image = cv2.blur(image, (3, 3))
                                image = cv2.resize(
                                    image,
                                    resize_shape,
                                    interpolation,
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
