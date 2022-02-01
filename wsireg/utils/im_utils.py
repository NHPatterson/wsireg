import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import SimpleITK as sitk
import zarr
from czifile import CziFile
from tifffile import (
    OmeXml,
    TiffFile,
    TiffWriter,
    create_output,
    imread,
    xml2dict,
)

from wsireg.parameter_maps.preprocessing import BoundingBox
from wsireg.utils.tform_utils import sitk_transform_image

TIFFFILE_EXTS = [".scn", ".tif", ".tiff", ".ndpi", ".svs"]

ARRAYLIKE_CLASSES = (np.ndarray, da.core.Array, zarr.Array)

SITK_TO_NP_DTYPE = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.int64,
    7: np.uint64,
    8: np.float32,
    9: np.float64,
    10: np.complex64,
    11: np.complex64,
    12: np.int8,
    13: np.uint8,
    14: np.int16,
    15: np.int16,
    16: np.int32,
    17: np.int32,
    18: np.int64,
    19: np.int64,
    20: np.float32,
    21: np.float64,
    22: np.uint8,
    23: np.uint16,
    24: np.uint32,
    25: np.uint64,
}

COLNAME_TO_HEX = {
    "red": "FF0000",
    "green": "00FF00",
    "blue": "0000FF",
    "magenta": "FF00FF",
    "yellow": "FFFF00",
    "cyan": "00FFFFF",
    "white": "FFFFFF",
}


def zarr_get_base_pyr_layer(zarr_store):
    """
    Find the base pyramid layer of a zarr store

    Parameters
    ----------
    zarr_store
        zarr store

    Returns
    -------
    zarr_im: zarr.core.Array
        zarr array of base layer
    """
    if isinstance(zarr_store, zarr.hierarchy.Group):
        zarr_im = zarr_store[str(0)]
    elif isinstance(zarr_store, zarr.core.Array):
        zarr_im = zarr_store
    return zarr_im


def ensure_dask_array(image):
    if isinstance(image, da.core.Array):
        return image

    if isinstance(image, zarr.Array):
        return da.from_zarr(image)

    # handles np.ndarray _and_ other array like objects.
    return da.from_array(image)


def read_preprocess_array(array, preprocessing, force_rgb=None):
    """Read np.array, zarr.Array, or dask.array image into memory
    with preprocessing for registration."""
    is_interleaved = guess_rgb(array.shape)
    is_rgb = is_interleaved if not force_rgb else force_rgb

    if is_rgb:
        if preprocessing:
            image_out = np.asarray(
                grayscale(array, is_interleaved=is_interleaved)
            )
            image_out = sitk.GetImageFromArray(image_out)
        else:
            image_out = np.asarray(array)
            if not is_interleaved:
                image_out = np.rollaxis(image_out, 0, 3)
            image_out = sitk.GetImageFromArray(image_out, isVector=True)

    elif len(array.shape) == 2:
        image_out = sitk.GetImageFromArray(np.asarray(array))

    else:
        if preprocessing:
            if preprocessing.ch_indices and len(array.shape) > 2:
                chs = list(preprocessing.ch_indices)
                array = array[chs, :, :]

        image_out = sitk.GetImageFromArray(np.squeeze(np.asarray(array)))

    return image_out


def tifffile_zarr_backend(
    image_filepath, largest_series, preprocessing, force_rgb=None
):
    """
    Read image with tifffile and use zarr to read data into memory

    Parameters
    ----------
    image_filepath: str
        path to the image file
    largest_series: int
        index of the largest series in the image
    preprocessing:
        whether to do some read-time pre-processing
        - greyscale conversion (at the tile level)
        - read individual or range of channels (at the tile level)

    Returns
    -------
    image: sitk.Image
        image ready for other registration pre-processing

    """
    print("using zarr backend")
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)
    zarr_im = zarr_get_base_pyr_layer(zarr_store)
    return read_preprocess_array(
        zarr_im, preprocessing=preprocessing, force_rgb=force_rgb
    )


def tifffile_dask_backend(
    image_filepath, largest_series, preprocessing, force_rgb=None
):
    """
    Read image with tifffile and use dask to read data into memory

    Parameters
    ----------
    image_filepath: str
        path to the image file
    largest_series: int
        index of the largest series in the image
    preprocessing:
        whether to do some read-time pre-processing
        - greyscale conversion (at the tile level)
        - read individual or range of channels (at the tile level)

    Returns
    -------
    image: sitk.Image
        image ready for other registration pre-processing

    """
    print("using dask backend")
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)
    dask_im = da.squeeze(da.from_zarr(zarr_get_base_pyr_layer(zarr_store)))
    return read_preprocess_array(
        dask_im, preprocessing=preprocessing, force_rgb=force_rgb
    )


def sitk_backend(image_filepath, preprocessing):
    """
    Read image with SimpleITK..this will always read the full image into memory

    Parameters
    ----------
    image_filepath: str
        path to the image file
    preprocessing:
        whether to do some read-time pre-processing
        - greyscale conversion (at the tile level)
        - read individual or range of channels (at the tile level)

    Returns
    -------
    image: sitk.Image
        image ready for other registration pre-processing

    """
    print("using sitk backend")
    image = sitk.ReadImage(image_filepath)

    if image.GetNumberOfComponentsPerPixel() >= 3:
        if preprocessing is not None:
            image = sitk_vect_to_gs(image)

    elif image.GetDepth() == 0:
        return image
    else:
        if preprocessing is not None:
            if (
                preprocessing.get("ch_indices") is not None
                and image.GetDepth() > 0
            ):
                print("here")
                chs = np.asarray(preprocessing.get('ch_indices'))
                image = image[:, :, chs]

    return image


def guess_rgb(shape):
    """
    Guess if the passed shape comes from rgb data.
    If last dim is 3 or 4 assume the data is rgb, including rgba.

    Parameters
    ----------
    shape : list of int
        Shape of the data that should be checked.

    Returns
    -------
    bool
        If data is rgb or not.
    """
    ndim = len(shape)
    last_dim = shape[-1]
    if ndim > 2 and last_dim < 5:
        rgb = True
    else:
        rgb = False

    return rgb


def grayscale(rgb_image, is_interleaved=False):
    """
    convert RGB image data to greyscale

    Parameters
    ----------
    rgb_image: np.ndarray
        image data
    Returns
    -------
    image:np.ndarray
        returns 8-bit greyscale image for 24-bit RGB image
    """
    if is_interleaved is True:
        result = (
            (rgb_image[..., 0] * 0.2125).astype(np.uint8)
            + (rgb_image[..., 1] * 0.7154).astype(np.uint8)
            + (rgb_image[..., 2] * 0.0721).astype(np.uint8)
        )
    else:
        result = (
            (rgb_image[0, ...] * 0.2125).astype(np.uint8)
            + (rgb_image[1, ...] * 0.7154).astype(np.uint8)
            + (rgb_image[2, ...] * 0.0721).astype(np.uint8)
        )

    return result


def czi_tile_grayscale(rgb_image):
    """
    convert RGB image data to greyscale

    Parameters
    ----------
    rgb_image: np.ndarray
        image data
    Returns
    -------
    image:np.ndarray
        returns 8-bit greyscale image for 24-bit RGB image
    """
    result = (
        (rgb_image[..., 0] * 0.2125).astype(np.uint8)
        + (rgb_image[..., 1] * 0.7154).astype(np.uint8)
        + (rgb_image[..., 2] * 0.0721).astype(np.uint8)
    )

    return np.expand_dims(result, axis=-1)


class CziRegImageReader(CziFile):
    """
    Sub-class of CziFile with added functionality to only read certain channels
    """

    def sub_asarray(
        self,
        resize=True,
        order=0,
        out=None,
        max_workers=None,
        channel_idx=None,
        as_uint8=False,
    ):

        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.
        channel_idx : int or list of int
            The indices of the channels to extract
        as_uint8 : bool
            byte-scale image data to np.uint8 data type

        Parameters
        ----------
        out:np.ndarray
            image read with selected parameters as np.ndarray
        """

        out_shape = list(self.shape)
        start = list(self.start)

        ch_dim_idx = self.axes.index('C')

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]

            if out_shape[ch_dim_idx] == 1:
                channel_idx = None

            else:
                out_shape[ch_dim_idx] = len(channel_idx)
                min_ch_seq = {}
                for idx, i in enumerate(channel_idx):
                    min_ch_seq.update({i: idx})

        if as_uint8 is True:
            out_dtype = np.uint8
        else:
            out_dtype = self.dtype

        if out is None:
            out = create_output(None, tuple(out_shape), out_dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        def func(
            directory_entry, resize=resize, order=order, start=start, out=out
        ):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            dvstart = list(directory_entry.start)
            czi_c_idx = [
                de.dimension for de in subblock.dimension_entries
            ].index('C')
            subblock_ch_idx = subblock.dimension_entries[czi_c_idx].start
            if channel_idx is not None:
                if subblock_ch_idx in channel_idx:
                    subblock.dimension_entries[czi_c_idx].start
                    tile = subblock.data(resize=resize, order=order)
                    dvstart[ch_dim_idx] = min_ch_seq.get(subblock_ch_idx)
                else:
                    return
            else:
                tile = subblock.data(resize=resize, order=order)

            if as_uint8 is True:
                tile = (tile / 256).astype("uint8")

            index = tuple(
                slice(i - j, i - j + k)
                for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape)
            )

            try:
                out[index] = tile
            except ValueError as e:
                warnings.warn(str(e))

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            for directory_entry in self.filtered_subblock_directory:
                func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out

    def sub_asarray_rgb(
        self,
        resize=True,
        order=0,
        out=None,
        max_workers=None,
        channel_idx=None,
        as_uint8=False,
        greyscale=False,
    ):

        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.
        channel_idx : int or list of int
            The indices of the channels to extract
        as_uint8 : bool
            byte-scale image data to np.uint8 data type

        Parameters
        ----------
        out:np.ndarray
            image read with selected parameters as np.ndarray
        """

        out_shape = list(self.shape)
        start = list(self.start)
        ch_dim_idx = self.axes.index('0')

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]
            out_shape[ch_dim_idx] = len(channel_idx)

        if greyscale is True:
            out_shape[ch_dim_idx] = 1

        if as_uint8 is True:
            out_dtype = np.uint8
        else:
            out_dtype = self.dtype

        if out is None:
            out = create_output(None, tuple(out_shape), out_dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        def func(
            directory_entry, resize=resize, order=order, start=start, out=out
        ):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            dvstart = list(directory_entry.start)
            tile = subblock.data(resize=resize, order=order)

            if greyscale is True:
                tile = czi_tile_grayscale(tile)

            if channel_idx is not None:
                tile = tile[:, :, :, :, :, channel_idx]

            index = tuple(
                slice(i - j, i - j + k)
                for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape)
            )

            try:
                out[index] = tile
            except ValueError as e:
                warnings.warn(str(e))

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            for directory_entry in self.filtered_subblock_directory:
                func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out


def tf_get_largest_series(image_filepath):
    """
    Determine largest series for .scn files by examining metadata
    For other multi-series files, find the one with the most pixels

    Parameters
    ----------
    image_filepath: str
        path to the image file

    Returns
    -------
    largest_series:int
        index of the largest series in the image data
    """
    fp_ext = Path(image_filepath).suffix.lower()
    tf_im = TiffFile(image_filepath)
    if fp_ext == ".scn":
        scn_meta = xml2dict(tf_im.scn_metadata)
        image_meta = scn_meta.get("scn").get("collection").get("image")
        largest_series = np.argmax(
            [
                im.get("scanSettings")
                .get("objectiveSettings")
                .get("objective")
                for im in image_meta
            ]
        )
    else:
        largest_series = np.argmax(
            [
                np.prod(np.asarray(series.shape), dtype=np.int64)
                for series in tf_im.series
            ]
        )
    return largest_series


def get_sitk_image_info(image_filepath):
    """
    Get image info for files only ready by SimpleITK

    Parameters
    ----------
    image_filepath:str
        filepath to image

    Returns
    -------
    im_dims: np.ndarray
        image dimensions in np.ndarray
    im_dtype: np.dtype
        data type of the image

    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_filepath)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    im_dims = np.asarray(reader.GetSize())
    # swap to YX
    im_dims[[0, 1]] = im_dims[[1, 0]]
    im_dtype = np.dtype(SITK_TO_NP_DTYPE.get(reader.GetPixelID()))
    is_vector = sitk.GetPixelIDValueAsString(reader.GetPixelID())

    if "vector" in is_vector:
        im_dims = np.append(im_dims, 3)
    elif len(im_dims) == 3:
        im_dims = im_dims[[2, 0, 1]]
    else:
        im_dims = np.concatenate([[1], im_dims], axis=0)

    return im_dims, im_dtype


def get_tifffile_info(image_filepath):
    largest_series = tf_get_largest_series(image_filepath)
    zarr_im = zarr.open(
        imread(image_filepath, aszarr=True, series=largest_series)
    )
    zarr_im = zarr_get_base_pyr_layer(zarr_im)
    im_dims = np.squeeze(zarr_im.shape)
    if len(im_dims) == 2:
        im_dims = np.concatenate([[1], im_dims])
    im_dtype = zarr_im.dtype

    return im_dims, im_dtype


def tf_zarr_read_single_ch(
    image_filepath, channel_idx, is_rgb, is_rgb_interleaved=True
):
    """
    Reads a single channel using zarr or dask in combination with tifffile

    Parameters
    ----------
    image_filepath:str
        file path to image
    channel_idx:int
        index of the channel to be read
    is_rgb:bool
        whether image is rgb interleaved

    Returns
    -------
    im:np.ndarray
        image as a np.ndarray
    """
    largest_series = tf_get_largest_series(image_filepath)
    zarr_im = zarr.open(
        imread(image_filepath, aszarr=True, series=largest_series)
    )
    zarr_im = zarr_get_base_pyr_layer(zarr_im)
    try:
        im = da.squeeze(da.from_zarr(zarr_im))
        if is_rgb and is_rgb_interleaved is True:
            im = im[:, :, channel_idx].compute()
        elif len(im.shape) > 2:
            im = im[channel_idx, :, :].compute()
        else:
            im = im.compute()

    except ValueError:
        im = zarr_im
        if is_rgb is True and is_rgb_interleaved is True:
            im = im[:, :, channel_idx]
        elif len(im.shape) > 2:
            im = im[channel_idx, :, :].compute()
        else:
            im = im.compute()
    return im


def czi_read_single_ch(image_filepath, channel_idx):
    czi = CziRegImageReader(image_filepath)

    im = czi.sub_asarray(
        channel_idx=channel_idx,
    )

    return im


def calc_pyramid_levels(xy_final_shape, tile_size):
    """
    Calculate number of pyramids for a given image dimension and tile size
    Stops when further downsampling would be smaller than tile_size.

    Parameters
    ----------
    xy_final_shape:np.ndarray
        final shape in xy order
    tile_size: int
        size of the tiles in the pyramidal layers
    Returns
    -------
    res_shapes:list
        list of tuples of the shapes of the downsampled images

    """
    res_shape = xy_final_shape[::-1]
    res_shapes = [tuple(res_shape)]

    while all(res_shape > tile_size):
        res_shape = res_shape // 2
        res_shapes.append(tuple(res_shape))

    return res_shapes[:-1]


def add_ome_axes_single_plane(image_np):
    """
    Reshapes np.ndarray image to match OME-zarr standard

    Parameters
    ----------
    image_np:np.ndarray
        image to which additional axes are added to meet OME-zarr standard

    Returns
    -------
    image_np:np.ndarray
        reshaped image array

    """
    return image_np.reshape((1,) * (3) + image_np.shape)


def generate_channels(channel_names, channel_colors, im_dtype):
    """
    Generate OME-zarr channels metadata

    Parameters
    ----------
    channel_names:list
    channel_colors:list
    im_dtype:np.dtype

    Returns
    -------
    channel_info:list
        list of dicts containing OME-zarr channel info
    """
    channel_info = []
    for channel_name, channel_color in zip(channel_names, channel_colors):
        channel_info.append(
            {
                "label": channel_name,
                "color": channel_color,
                "active": True,
                "window": {"start": 0, "end": int(np.iinfo(im_dtype).max)},
            }
        )
    return channel_info


def format_channel_names(channel_names, n_ch):
    """
    Format channel names and ensure number of channel names matches number of channels or default
    to C1, C2, C3, etc.

    Parameters
    ----------
    channel_names:list
        list of str that are channel names
    n_ch: int
        number of channels detected in image

    Returns
    -------
    channel_names:
        list of str that are formatted
    """
    if channel_names is None or n_ch != len(channel_names):
        channel_names = ["C{}".format(idx) for idx in range(n_ch)]
    return channel_names


def get_pyramid_info(y_size, x_size, n_ch, tile_size):
    """
    Get pyramidal info for OME-zarr output

    Parameters
    ----------
    y_size: int
        y dimension of base layer
    x_size:int
        x dimension of base layer
    n_ch:int
        number of channels in the image
    tile_size:int
        tile size of the image

    Returns
    -------
    pyr_levels
        pyramidal levels
    pyr_shapes:
        OME-zarr pyramid shapes for all levels

    """
    yx_size = np.asarray([y_size, x_size], dtype=np.int32)
    pyr_levels = calc_pyramid_levels(yx_size, tile_size)
    pyr_shapes = [(1, n_ch, 1, int(pl[0]), int(pl[1])) for pl in pyr_levels]
    return pyr_levels, pyr_shapes


def prepare_ome_zarr_group(
    zarr_store_dir,
    y_size,
    x_size,
    n_ch,
    im_dtype,
    tile_size=512,
    channel_names=None,
    channel_colors=None,
):
    """
    Prepare OME-zarr store with all meta data and channel info and initialize store

    Parameters
    ----------
    zarr_store_dir:str
        filepath to zarr store
    y_size: int
        y dimension of base layer
    x_size:int
        x dimension of base layer
    n_ch:int
        number of channels in the image
    im_dtype:np.dtype
        data type of the image
    tile_size:int
        tile size of the image
    channel_names:list
        list of str channel names
    channel_colors:
        list of hex or str channel colors

    Returns
    -------
    grp: zarr.store
        initialized store
    n_pyr_levels: int
        number of sub-resolutions
    pyr_levels: list
        shapes of sub-resolutions
    """
    store = zarr.DirectoryStore(zarr_store_dir)
    grp = zarr.group(store, overwrite=True)
    zarr_dtype = "{}{}".format(im_dtype.kind, im_dtype.itemsize)

    pyr_levels, pyr_shapes = get_pyramid_info(x_size, y_size, n_ch, tile_size)

    paths = []
    for path, pyr_shape in enumerate(pyr_levels):
        grp.create_dataset(
            str(path),
            shape=pyr_shapes[path],
            dtype=zarr_dtype,
            chunks=(1, 1, 1, tile_size, tile_size),
        )
        paths.append({"path": str(path)})

    multiscales = [
        {
            "version": "0.1",
            "datasets": paths,
        }
    ]
    grp.attrs["multiscales"] = multiscales
    n_pyr_levels = len(paths)

    channel_names = format_channel_names(channel_names, n_ch)

    n_colors = n_ch // len(COLNAME_TO_HEX) + 1
    color_palette = [*COLNAME_TO_HEX] * n_colors

    if channel_colors is None:
        channel_colors = [color_palette[idx] for idx in range(n_ch)]
    elif n_ch != len(channel_colors) and n_ch != 1:
        channel_colors = [color_palette[idx] for idx in range(n_ch)]
    elif n_ch != len(channel_colors) and n_ch == 1:
        channel_colors = ["FFFFFF"]
    else:
        channel_colors = [COLNAME_TO_HEX[ch] for ch in channel_colors]

    channel_info = generate_channels(channel_names, channel_colors, im_dtype)

    image_data = {
        'id': 1,
        'channels': channel_info,
        'rdefs': {
            'model': 'color',
        },
    }

    grp.attrs["omero"] = image_data

    return grp, n_pyr_levels, pyr_levels


def get_final_tform_info(final_transform):
    """
    Extract size and spacing information from wsireg's final transformation elastix data

    Parameters
    ----------
    final_transform:itk.Transform
        itk.Transform with added attributes containing transform data

    Returns
    -------
    y_size: int
    x_size: int
    y_spacing: float
    x_spacing: float

    """
    x_size, y_size = (
        final_transform.output_size[0],
        final_transform.output_size[1],
    )
    x_spacing, y_spacing = (
        final_transform.output_spacing[0],
        final_transform.output_spacing[1],
    )
    return int(y_size), int(x_size), float(y_spacing), float(x_spacing)


def image_to_zarr_store(zgrp, image, channel_idx, n_pyr_levels, pyr_levels):
    """
    Write image into zarr store with sub resolutions

    Parameters
    ----------
    grp: zarr.store
        initialized store
    image:sitk.Image
        image
    channel_idx:int
        which channel the image represents
    n_pyr_levels: int
    pyr_levels: list
    """
    for pyr_idx in range(n_pyr_levels):
        if pyr_idx == 0:
            image = sitk.GetArrayFromImage(image)
        else:
            resize_shape = (
                pyr_levels[pyr_idx][1],
                pyr_levels[pyr_idx][0],
            )
            image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)

        zgrp[str(pyr_idx)][
            :, channel_idx : channel_idx + 1, :, :, :
        ] = add_ome_axes_single_plane(image)


def prepare_ome_xml_str(
    y_size, x_size, n_ch, im_dtype, is_rgb, **ome_metadata
):

    omexml = OmeXml()
    if is_rgb:
        stored_shape = (1, 1, 1, y_size, x_size, n_ch)
        im_shape = (y_size, x_size, n_ch)
    else:
        stored_shape = (n_ch, 1, 1, y_size, x_size, 1)
        im_shape = (n_ch, y_size, x_size)

    omexml.addimage(
        dtype=im_dtype,
        shape=im_shape,
        # specify how the image is stored in the TIFF file
        storedshape=stored_shape,
        **ome_metadata,
    )

    return omexml.tostring().encode("utf8")


def get_final_yx_from_tform(tform_reg_im, final_transform):
    if final_transform is not None:
        y_size, x_size, y_spacing, x_spacing = get_final_tform_info(
            final_transform
        )
    else:
        y_size, x_size = (
            (tform_reg_im.im_dims[0], tform_reg_im.im_dims[1])
            if tform_reg_im.is_rgb
            else (tform_reg_im.im_dims[1], tform_reg_im.im_dims[2])
        )
        y_spacing, x_spacing = None, None
    return y_size, x_size, y_spacing, x_spacing


def transform_to_ome_zarr(tform_reg_im, output_dir, tile_size=512):

    y_size, x_size = get_final_yx_from_tform(tform_reg_im)

    n_ch = (
        tform_reg_im.im_dims[2]
        if tform_reg_im.is_rgb
        else tform_reg_im.im_dims[0]
    )
    pyr_levels, pyr_shapes = get_pyramid_info(y_size, x_size, n_ch, tile_size)
    n_pyr_levels = len(pyr_levels)
    output_file_name = str(Path(output_dir) / tform_reg_im.image_name)
    if tform_reg_im.reader in ["tifffile", "czi", "sitk"]:
        if tform_reg_im.reader == "sitk":
            full_image = sitk.ReadImage(tform_reg_im.image_filepath)
        for channel_idx in range(n_ch):
            if tform_reg_im.reader == "tifffile":
                tform_reg_im.image = tf_zarr_read_single_ch(
                    tform_reg_im.image_filepath,
                    channel_idx,
                    tform_reg_im.is_rgb,
                )
                tform_reg_im.image = np.squeeze(tform_reg_im.image)

            elif tform_reg_im.reader == "czi":
                tform_reg_im.image = czi_read_single_ch(
                    tform_reg_im.image_filepath, channel_idx
                )
                tform_reg_im.image = np.squeeze(tform_reg_im.image)
            elif tform_reg_im.reader == "sitk":
                if tform_reg_im.is_rgb:
                    tform_reg_im.image = sitk.VectorIndexSelectionCast(
                        full_image
                    )
                elif len(full_image.GetSize()) > 2:
                    tform_reg_im.image = full_image[:, :, channel_idx]
                else:
                    tform_reg_im.image = full_image

            if tform_reg_im.composite_transform is not None:
                tform_reg_im = transform_plane(tform_reg_im)
            else:
                tform_reg_im.image = sitk.GetImageFromArray(tform_reg_im.image)
            if channel_idx == 0:
                channel_names = format_channel_names(
                    tform_reg_im.channel_names, n_ch
                )
                print(f"saving to {output_file_name}.ome.zarr")

                (zgrp, n_pyr_levels, pyr_levels,) = prepare_ome_zarr_group(
                    f"{output_file_name}.ome.zarr",
                    y_size,
                    x_size,
                    n_ch,
                    tform_reg_im.im_dtype,
                    tile_size=tile_size,
                    channel_names=channel_names,
                    channel_colors=tform_reg_im.channel_colors,
                )

            image_to_zarr_store(
                zgrp, tform_reg_im.image, channel_idx, n_pyr_levels, pyr_levels
            )

    return f"{output_file_name}.ome.zarr"


def transform_plane(image, final_transform, composite_transform):

    image = sitk_transform_image(
        image,
        final_transform,
        composite_transform,
    )

    return image


def transform_to_ome_tiff(
    tform_reg_im,
    image_name,
    output_dir,
    final_transform,
    composite_transform,
    tile_size=512,
    write_pyramid=True,
):

    y_size, x_size, y_spacing, x_spacing = get_final_yx_from_tform(
        tform_reg_im, final_transform
    )

    # protect against too large tile size
    while y_size / tile_size <= 1 or x_size / tile_size <= 1:
        tile_size = tile_size // 2

    n_ch = (
        tform_reg_im.im_dims[2]
        if tform_reg_im.is_rgb
        else tform_reg_im.im_dims[0]
    )
    pyr_levels, pyr_shapes = get_pyramid_info(y_size, x_size, n_ch, tile_size)
    n_pyr_levels = len(pyr_levels)
    output_file_name = str(Path(output_dir) / image_name)
    channel_names = format_channel_names(tform_reg_im.channel_names, n_ch)

    if final_transform is not None:
        PhysicalSizeY = y_spacing
        PhysicalSizeX = x_spacing
    else:
        PhysicalSizeY = tform_reg_im.image_res
        PhysicalSizeX = tform_reg_im.image_res

    omexml = prepare_ome_xml_str(
        y_size,
        x_size,
        n_ch,
        tform_reg_im.im_dtype,
        tform_reg_im.is_rgb,
        PhysicalSizeX=PhysicalSizeX,
        PhysicalSizeY=PhysicalSizeY,
        PhysicalSizeXUnit="µm",
        PhysicalSizeYUnit="µm",
        Name=image_name,
        Channel=None if tform_reg_im.is_rgb else {"Name": channel_names},
    )
    subifds = n_pyr_levels - 1 if write_pyramid is True else None

    rgb_im_data = []

    if tform_reg_im.reader == "sitk":
        full_image = sitk.ReadImage(tform_reg_im.image_filepath)

    print(f"saving to {output_file_name}.ome.tiff")
    with TiffWriter(f"{output_file_name}.ome.tiff", bigtiff=True) as tif:
        for channel_idx in range(n_ch):
            print(f"transforming : {channel_idx}")
            if tform_reg_im.reader != "sitk":
                image = tform_reg_im.read_single_channel(channel_idx)
                image = np.squeeze(image)
                image = sitk.GetImageFromArray(image)
                image.SetSpacing(
                    (tform_reg_im.image_res, tform_reg_im.image_res)
                )
            else:
                if tform_reg_im.is_rgb:
                    image = sitk.VectorIndexSelectionCast(
                        full_image, channel_idx
                    )
                elif len(full_image.GetSize()) > 2:
                    image = full_image[:, :, channel_idx]
                else:
                    image = full_image

            if composite_transform is not None:
                image = transform_plane(
                    image, final_transform, composite_transform
                )
                print(f"transformed : {channel_idx}")

            if tform_reg_im.is_rgb:
                rgb_im_data.append(image)
            else:
                print("saving")
                if isinstance(image, sitk.Image):
                    image = sitk.GetArrayFromImage(image)

                options = dict(
                    tile=(tile_size, tile_size),
                    compression="jpeg" if tform_reg_im.is_rgb else "deflate",
                    photometric="rgb" if tform_reg_im.is_rgb else "minisblack",
                    metadata=None,
                )
                # write OME-XML to the ImageDescription tag of the first page
                description = omexml if channel_idx == 0 else None
                # write channel data
                print(f" writing channel {channel_idx} - shape: {image.shape}")
                tif.write(
                    image,
                    subifds=subifds,
                    description=description,
                    **options,
                )

                if write_pyramid:
                    for pyr_idx in range(1, n_pyr_levels):
                        resize_shape = (
                            pyr_levels[pyr_idx][0],
                            pyr_levels[pyr_idx][1],
                        )
                        image = cv2.resize(
                            image,
                            resize_shape,
                            cv2.INTER_LINEAR,
                        )
                        print(
                            f"pyr {pyr_idx} : channel {channel_idx} shape: {image.shape}"
                        )

                        tif.write(image, **options, subfiletype=1)

        if tform_reg_im.is_rgb:
            rgb_im_data = sitk.Compose(rgb_im_data)
            rgb_im_data = sitk.GetArrayFromImage(rgb_im_data)

            options = dict(
                tile=(tile_size, tile_size),
                compression="jpeg" if tform_reg_im.is_rgb else None,
                photometric="rgb" if tform_reg_im.is_rgb else "minisblack",
                metadata=None,
            )
            # write OME-XML to the ImageDescription tag of the first page
            description = omexml

            # write channel data
            tif.write(
                rgb_im_data,
                subifds=subifds,
                description=description,
                **options,
            )

            print(f"RGB shape: {rgb_im_data.shape}")
            if write_pyramid:
                for pyr_idx in range(1, n_pyr_levels):
                    resize_shape = (
                        pyr_levels[pyr_idx][0],
                        pyr_levels[pyr_idx][1],
                    )
                    rgb_im_data = cv2.resize(
                        rgb_im_data, resize_shape, cv2.INTER_LINEAR
                    )
                    print(f"pyr {pyr_idx} : RGB , shape: {rgb_im_data.shape}")

                    tif.write(rgb_im_data, **options, subfiletype=1)

    return f"{output_file_name}.ome.tiff"


def transform_to_ome_tiff_merge(
    tform_reg_im,
    image_name,
    output_dir,
    final_transform,
    composite_transform,
    tile_size=512,
    write_pyramid=True,
):

    y_size, x_size, y_spacing, x_spacing = get_final_yx_from_tform(
        tform_reg_im.images[0], final_transform[0]
    )

    # protect against too large tile size
    while y_size / tile_size <= 1 or x_size / tile_size <= 1:
        tile_size = tile_size // 2

    n_ch = tform_reg_im.n_ch
    pyr_levels, pyr_shapes = get_pyramid_info(y_size, x_size, n_ch, tile_size)
    n_pyr_levels = len(pyr_levels)
    output_file_name = str(Path(output_dir) / image_name)
    channel_names = format_channel_names(tform_reg_im.channel_names, n_ch)

    if final_transform is not None:
        PhysicalSizeY = y_spacing
        PhysicalSizeX = x_spacing
    else:
        PhysicalSizeY = tform_reg_im.image_res
        PhysicalSizeX = tform_reg_im.image_res

    omexml = prepare_ome_xml_str(
        y_size,
        x_size,
        n_ch,
        tform_reg_im.images[0].im_dtype,
        tform_reg_im.images[0].is_rgb,
        PhysicalSizeX=PhysicalSizeX,
        PhysicalSizeY=PhysicalSizeY,
        PhysicalSizeXUnit="µm",
        PhysicalSizeYUnit="µm",
        Name=image_name,
        Channel={"Name": channel_names},
    )
    subifds = n_pyr_levels - 1 if write_pyramid is True else None

    print(f"saving to {output_file_name}.ome.tiff")
    with TiffWriter(f"{output_file_name}.ome.tiff", bigtiff=True) as tif:
        for m_idx, merge_image in enumerate(tform_reg_im.images):
            merge_n_ch = merge_image.n_ch
            for channel_idx in range(merge_n_ch):
                image = merge_image.read_single_channel(channel_idx)
                image = np.squeeze(image)
                image = sitk.GetImageFromArray(image)
                image.SetSpacing(
                    (merge_image.image_res, merge_image.image_res)
                )

                if composite_transform[m_idx] is not None:
                    image = transform_plane(
                        image,
                        final_transform[m_idx],
                        composite_transform[m_idx],
                    )

                print("saving")
                if isinstance(image, sitk.Image):
                    image = sitk.GetArrayFromImage(image)

                options = dict(
                    tile=(tile_size, tile_size),
                    compression="jpeg" if merge_image.is_rgb else "deflate",
                    photometric="rgb" if merge_image.is_rgb else "minisblack",
                    metadata=None,
                )
                # write OME-XML to the ImageDescription tag of the first page
                description = omexml if channel_idx == 0 else None
                # write channel data
                print(f" writing channel {channel_idx} - shape: {image.shape}")
                tif.write(
                    image,
                    subifds=subifds,
                    description=description,
                    **options,
                )

                if write_pyramid:
                    for pyr_idx in range(1, n_pyr_levels):
                        resize_shape = (
                            pyr_levels[pyr_idx][0],
                            pyr_levels[pyr_idx][1],
                        )
                        image = cv2.resize(
                            image,
                            resize_shape,
                            cv2.INTER_LINEAR,
                        )
                        print(
                            f"pyr {pyr_idx} : channel {channel_idx} shape: {image.shape}"
                        )

                        tif.write(image, **options, subfiletype=1)

        return f"{output_file_name}.ome.tiff"


def compute_mask_to_bbox(mask, mask_padding=100):
    mask.SetSpacing((1, 1))
    mask_size = mask.GetSize()
    mask = sitk.Threshold(mask, 1, 255)
    mask = sitk.ConnectedComponent(mask)

    labstats = sitk.LabelShapeStatisticsImageFilter()
    labstats.SetBackgroundValue(0)
    labstats.ComputePerimeterOff()
    labstats.ComputeFeretDiameterOff()
    labstats.ComputeOrientedBoundingBoxOff()
    labstats.Execute(mask)

    bb_points = []
    for label in labstats.GetLabels():
        x1, y1, xw, yh = labstats.GetBoundingBox(label)
        x2, y2 = x1 + xw, y1 + yh
        lab_points = np.asarray([[x1, y1], [x2, y2]])
        bb_points.append(lab_points)

    bb_points = np.concatenate(bb_points)
    x_min = np.min(bb_points[:, 0])
    y_min = np.min(bb_points[:, 1])
    x_max = np.max(bb_points[:, 0])
    y_max = np.max(bb_points[:, 1])

    if (x_min - mask_padding) < 0:
        x_min = 0
    else:
        x_min -= mask_padding

    if (y_min - mask_padding) < 0:
        y_min = 0
    else:
        y_min -= mask_padding

    if (x_max + mask_padding) > mask_size[0]:
        x_max = mask_size[0]
    else:
        x_max += mask_padding

    if (y_max + mask_padding) > mask_size[1]:
        y_max = mask_size[1]
    else:
        y_max += mask_padding

    x_width = x_max - x_min
    y_height = y_max - y_min

    return BoundingBox(x_min, y_min, x_width, y_height)


def sitk_vect_to_gs(image):
    """
    converts simpleITK RGB image to greyscale using cv2
    Parameters
    ----------
    image
        SimpleITK image

    Returns
    -------
        Greyscale SimpleITK image
    """
    image = sitk.GetArrayFromImage(image)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return sitk.GetImageFromArray(image, isVector=False)


def sitk_max_int_proj(image):
    """
    Finds maximum intensity projection of multi-channel SimpleITK image

    Parameters
    ----------
    image
        multichannel impleITK image


    Returns
    -------
    SimpleITK image
    """
    # check if there are 3 dimensions (XYC)
    if len(image.GetSize()) == 3:
        return sitk.MaximumProjection(image, 2)[:, :, 0]
    else:
        print(
            'cannot perform maximum intensity project on single channel image'
        )
        return image


def sitk_inv_int(image):
    """
    inverts intensity of images for registration, useful for alignment of brightfield
    and fluorescence images
    Parameters
    ----------
    image
        SimpleITK image s

    Returns
    -------
    Intensity inverted SimpleITK image
    """
    return sitk.InvertIntensity(image)


def contrast_enhance(image):
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.convertScaleAbs(image, alpha=7, beta=1)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def std_prepro():
    """
    Catch all dictionary of preprocessing functions that will result in a single 2D image for
    registration
    Returns
    -------
    dictionary of processing parameters
    """
    STD_PREPRO = {
        'image_type': 'FL',
        'ch_indices': None,
        'as_uint8': True,
        'downsample': 1,
        'max_int_proj': sitk_max_int_proj,
        'inv_int': sitk_inv_int,
    }
    return STD_PREPRO


def tifffile_to_arraylike(image_filepath):
    largest_series_idx = tf_get_largest_series(image_filepath)
    image = zarr.open(
        imread(image_filepath, aszarr=True, series=largest_series_idx)
    )
    if isinstance(image, zarr.Group):
        image = image[0]

    return image, image_filepath


def ome_tifffile_to_arraylike(image_filepath):
    ome_metadata = xml2dict(TiffFile(image_filepath).ome_metadata)
    im_dims, im_dtype = get_tifffile_info(image_filepath)

    largest_series_idx = tf_get_largest_series(image_filepath)

    series_metadata = ome_metadata.get("OME").get("Image")

    if isinstance(series_metadata, list):
        series_metadata = series_metadata[largest_series_idx]

    if isinstance(series_metadata.get("Pixels").get("Channel"), list):
        samples_per_pixel = (
            series_metadata.get("Pixels")
            .get("Channel")[0]
            .get("SamplesPerPixel")
        )
    else:
        samples_per_pixel = (
            series_metadata.get("Pixels").get("Channel").get("SamplesPerPixel")
        )

    is_rgb = guess_rgb(im_dims)

    image = zarr.open(
        imread(image_filepath, aszarr=True, series=largest_series_idx)
    )

    if isinstance(image, zarr.Group):
        image = image[0]

    image = da.from_zarr(image)
    if samples_per_pixel:
        if is_rgb is False and samples_per_pixel >= 3:
            image = image.transpose(1, 2, 0)

    return image, image_filepath


#
#
# def prepare_np_image(image, preprocessing):
#     """
#     preprocess images stored as numpy arrays
#
#     Parameters
#     ----------
#     image:np.ndarray
#         image data
#     preprocessing:dict
#         whether to do some read-time pre-processing
#         - greyscale conversion (at the tile level)
#         - read individual or range of channels (at the tile level)
#
#     Returns
#     -------
#     image:sitk.Image
#         image ready for registration
#     """
#     is_rgb = guess_rgb(image.shape)
#
#     # greyscale an RGB np array
#     if preprocessing is not None:
#         if is_rgb:
#             image = grayscale(image)
#             is_rgb = False
#
#         # select channels for registration if multi-channel
#         if image.ndim > 2 and preprocessing.get("ch_indices") is not None:
#             image = np.squeeze(image[preprocessing.get("ch_indices"), :, :])
#
#     # pass nparray to sitk image
#     image = sitk.GetImageFromArray(image, isVector=is_rgb)
#
#     # sitk preprocessing
#     if (
#         preprocessing is not None
#         and preprocessing.get('as_uint8') is True
#         and image.GetPixelID() != sitk.sitkUInt8
#     ):
#         image = sitk.RescaleIntensity(image)
#         image = sitk.Cast(image, sitk.sitkUInt8)
#
#     return image
#
#
# def read_image(image_filepath, preprocessing):
#     """
#     Convenience function to read images
#
#     Parameters
#     ----------
#     image_filepath : str
#         file path to image
#     preprocessing
#         read time preprocessing dict ("as_uint8" and "ch_indices")
#     Returns
#     -------
#     image: sitk.Image
#         SimpleITK image of image data from czi, scn or other image formats read by SimpleITK
#     """
#
#     fp_ext = Path(image_filepath).suffix.lower()
#
#     if fp_ext == '.czi':
#         czi = CziRegImageReader(image_filepath)
#         scene_idx = czi.axes.index('S')
#
#         if czi.shape[scene_idx] > 1:
#             raise ValueError('multi scene czis not allowed at this time')
#
#         if preprocessing is None:
#             image = czi.asarray()
#         else:
#             image = czi.sub_asarray(
#                 channel_idx=preprocessing['ch_indices'],
#                 as_uint8=preprocessing['as_uint8'],
#             )
#         image = np.squeeze(image)
#
#         image = sitk.GetImageFromArray(image)
#
#     # find out other WSI formats read by tifffile
#     elif fp_ext in TIFFFILE_EXTS:
#         largest_series = tf_get_largest_series(image_filepath)
#
#         try:
#             image = tifffile_dask_backend(
#                 image_filepath, largest_series, preprocessing
#             )
#         except ValueError:
#             image = tifffile_zarr_backend(
#                 image_filepath, largest_series, preprocessing
#             )
#
#         if (
#             preprocessing is not None
#             and preprocessing.get('as_uint8') is True
#             and image.GetPixelID() != sitk.sitkUInt8
#         ):
#             image = sitk.RescaleIntensity(image)
#             image = sitk.Cast(image, sitk.sitkUInt8)
#     else:
#         image = sitk_backend(image_filepath, preprocessing)
#
#     return image

#
# def get_im_info(image_filepath):
#     """
#     Use CziFile and tifffile to get image dimension and other information.
#
#     Parameters
#     ----------
#     image_filepath: str
#         filepath to the image file
#
#     Returns
#     -------
#     im_dims: np.ndarray
#         image dimensions in np.ndarray
#     im_dtype: np.dtype
#         data type of the image
#     reader:str
#         whether to use "czifile" or "tifffile" to read the image
#     """
#     if Path(image_filepath).suffix.lower() == ".czi":
#         czi = CziFile(image_filepath)
#         ch_dim_idx = czi.axes.index('C')
#         y_dim_idx = czi.axes.index('Y')
#         x_dim_idx = czi.axes.index('X')
#         im_dims = np.array(czi.shape)[[ch_dim_idx, y_dim_idx, x_dim_idx]]
#         im_dtype = czi.dtype
#         reader = "czi"
#
#     elif Path(image_filepath).suffix.lower() in TIFFFILE_EXTS:
#         largest_series = tf_get_largest_series(image_filepath)
#         zarr_im = zarr.open(
#             imread(image_filepath, aszarr=True, series=largest_series)
#         )
#         zarr_im = zarr_get_base_pyr_layer(zarr_im)
#         im_dims = np.squeeze(zarr_im.shape)
#         if len(im_dims) == 2:
#             im_dims = np.concatenate([[1], im_dims])
#         im_dtype = zarr_im.dtype
#         reader = "tifffile"
#
#     else:
#         im_dims, im_dtype = get_sitk_image_info(image_filepath)
#         reader = "sitk"
#
#     return im_dims, im_dtype, reader
