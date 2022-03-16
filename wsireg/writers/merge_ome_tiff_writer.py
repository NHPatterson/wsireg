from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk
from tifffile import TiffWriter

from wsireg.reg_images.reg_image import RegImage
from wsireg.reg_images.merge_reg_image import MergeRegImage
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
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
        reg_transform_seqs: Optional[List[RegTransformSeq]] = None,
    ):
        """
        Class for writing multiple images wiuth and without transforms to a singel OME-TIFF.

        Parameters
        ----------
        reg_image: MergeRegImage
            MergeRegImage to be transformed
        reg_transform_seqs: List of RegTransformSeq or None
            Registration transformation sequences for each wsireg image to be merged

        Attibutes
        ---------
        x_size: int
            Size of the merged image after transformation in x
        y_size: int
            Size of the merged image after transformation in y
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
        self.reg_image = reg_images
        self.reg_transform_seqs = reg_transform_seqs

    def _length_checks(self, sub_image_names):
        """Make sure incoming data is kosher in dimensions"""
        if isinstance(sub_image_names, list) is False:
            if sub_image_names is None:
                sub_image_names = [
                    "" for i in range(len(self.reg_image.images))
                ]
            else:
                raise ValueError(
                    "MergeRegImage requires a list of image names for each image to merge"
                )

        if self.reg_transform_seqs is None:
            transformations = [None for i in range(len(self.reg_image.images))]
        else:
            transformations = self.reg_transform_seqs

        if len(transformations) != len(self.reg_image.images):
            raise ValueError(
                "MergeRegImage number of transforms does not match number of images"
            )

    def _create_channel_names(self, sub_image_names):
        """Create channel names for merge data."""

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
        """Check that all transforms as currently loaded output to the same size/resolution"""
        out_size = []
        out_spacing = []

        if not self.reg_transform_seqs:
            rts = [None for _ in range(len(self.reg_image.images))]
        else:
            rts = self.reg_transform_seqs
        for im, t in zip(self.reg_image.images, rts):
            if t:
                out_size.append(t.reg_transforms[-1].output_size)
                out_spacing.append(t.reg_transforms[-1].output_spacing)
            else:
                out_im_size = (
                    (im.shape[0], im.shape[1])
                    if im.is_rgb
                    else (im.shape[1], im.shape[2])
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
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str = "default",
    ):
        """Prepare OME-XML and other data needed for saving"""

        if reg_transform_seq:
            self.x_size, self.y_size = reg_transform_seq.output_size
            self.x_spacing, self.y_spacing = reg_transform_seq.output_spacing
        else:
            self.y_size, self.x_size = (
                (reg_image.shape[0], reg_image.shape[1])
                if reg_image.is_rgb
                else (reg_image.shape[1], reg_image.shape[2])
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
        """Determine data type for merger. Will default to the largest
        dtype. If one image is np.uint8 and another np.uint16, the image at np.uint8
        will be cast to np.uint16"""
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
        image_name: str,
        sub_image_names: List[str],
        output_dir: Union[Path, str] = "",
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: Optional[str] = "default",
    ) -> str:
        """
         Write merged OME-TIFF image plane-by-plane to disk.
         RGB images will be de-interleaved with RGB channels written as separate planes.

         Parameters
         ----------
         image_name: str
             Name to be written WITHOUT extension
             for example if image_name = "cool_image" the file
             would be "cool_image.ome.tiff"
        sub_image_names: list of str
            Names added before each channel of a given image to distinguish it.
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
        merge_dtype_sitk, merge_dtype_np = self._get_merge_dtype()

        self._length_checks(sub_image_names)
        self._create_channel_names(sub_image_names)
        self._transform_check()

        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")

        self._prepare_image_info(
            self.reg_image.images[0],
            image_name,
            merge_dtype_np,
            reg_transform_seq=self.reg_transform_seqs[0],
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

                    if self.reg_transform_seqs[m_idx]:
                        image = self.reg_transform_seqs[
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
