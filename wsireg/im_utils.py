from pathlib import Path
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import SimpleITK as sitk
from tifffile import (
    create_output,
    TiffFile,
    xml2dict,
    TiffWriter,
    imread,
    OmeXml,
)
from czifile import CziFile
import zarr
import dask.array as da
from wsireg.reg_utils import apply_transform_dict

TIFFFILE_EXTS = [".scn", ".tif", ".tiff", ".ndpi"]

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
    if isinstance(zarr_store, zarr.hierarchy.Group):
        zarr_im = zarr_store[str(0)]
    elif isinstance(zarr_store, zarr.core.Array):
        zarr_im = zarr_store
    return zarr_im


def tifffile_zarr_backend(image_filepath, largest_series, preprocessing):

    print("using zarr backend")
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)
    zarr_im = zarr_get_base_pyr_layer(zarr_store)

    if guess_rgb(zarr_im.shape):
        if preprocessing is not None:
            image = grayscale(zarr_im)
            image = sitk.GetImageFromArray(image)
        else:
            image = zarr_im[:]
            image = sitk.GetImageFromArray(image, isVector=True)

    elif len(zarr_im.shape) == 2:
        image = sitk.GetImageFromArray(zarr_im[:])

    else:
        if preprocessing is not None:
            if (
                preprocessing.get("ch_indices") is not None
                and len(zarr_im.shape) > 2
            ):
                chs = tuple(preprocessing.get('ch_indices'))
                zarr_im = np.squeeze(zarr_im[:])
                zarr_im = zarr_im[chs, :, :]

        image = sitk.GetImageFromArray(np.squeeze(zarr_im[:]))

    return image


def tifffile_dask_backend(image_filepath, largest_series, preprocessing):
    print("using dask backend")
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)

    dask_im = da.squeeze(da.from_zarr(zarr_get_base_pyr_layer(zarr_store)))

    if guess_rgb(dask_im.shape):
        if preprocessing is not None:
            image = grayscale(dask_im).compute()
            image = sitk.GetImageFromArray(image)
        else:
            image = dask_im.compute()
            image = sitk.GetImageFromArray(image, isVector=True)

    elif len(dask_im.shape) == 2:
        image = sitk.GetImageFromArray(dask_im.compute())

    else:
        if preprocessing is not None:
            if (
                preprocessing.get("ch_indices") is not None
                and len(dask_im.shape) > 2
            ):
                chs = np.asarray(preprocessing.get('ch_indices'))
                dask_im = dask_im[chs, :, :]

        image = sitk.GetImageFromArray(np.squeeze(dask_im.compute()))

    return image


def sitk_backend(image_filepath, preprocessing):
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
                chs = np.asarray(preprocessing.get('ch_indices'))
                image = image[:, :, chs]

    return image


def guess_rgb(shape):
    """from: napari's internal im utils
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

    return ndim > 2 and last_dim < 5


def grayscale(rgb):
    result = (
        (rgb[..., 0] * 0.2125)
        + (rgb[..., 1] * 0.7154)
        + (rgb[..., 2] * 0.0721)
    )

    return result.astype(np.uint8)


class CziRegImageReader(CziFile):
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

        """

        out_shape = list(self.shape)
        start = list(self.start)

        ch_dim_idx = self.axes.index('C')

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]
            out_shape[ch_dim_idx] = len(channel_idx)
            min_ch = np.min(channel_idx)
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

            if channel_idx is not None:
                if subblock.dimension_entries[czi_c_idx].start in channel_idx:
                    tile = subblock.data(resize=resize, order=order)
                    dvstart[ch_dim_idx] = (
                        subblock.dimension_entries[czi_c_idx].start - min_ch
                    )
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


def tf_get_largest_series(image_filepath):
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
            [np.prod(np.asarray(series.shape)) for series in tf_im.series]
        )
    return largest_series


def read_image(image_filepath, preprocessing):
    """

    Parameters
    ----------
    image_filepath : str
        file path to image
    preprocessing
        read time preprocessing dict ("as_uint8" and "ch_indices")
    Returns
    -------
        SimpleITK image of image data from czi, scn or other image formats read by SimpleITK
    """

    fp_ext = Path(image_filepath).suffix.lower()

    if fp_ext == '.czi':
        czi = CziRegImageReader(image_filepath)
        scene_idx = czi.axes.index('S')

        if czi.shape[scene_idx] > 1:
            raise ValueError('multi scene czis not allowed at this time')

        if preprocessing is None:
            image = czi.asarray()
        else:
            image = czi.sub_asarray(
                channel_idx=preprocessing['ch_indices'],
                as_uint8=preprocessing['as_uint8'],
            )
        image = np.squeeze(image)

        image = sitk.GetImageFromArray(image)

    # find out other WSI formats read by tifffile
    elif fp_ext in TIFFFILE_EXTS:
        largest_series = tf_get_largest_series(image_filepath)

        try:
            image = tifffile_dask_backend(
                image_filepath, largest_series, preprocessing
            )
        except ValueError:
            image = tifffile_zarr_backend(
                image_filepath, largest_series, preprocessing
            )

        if (
            preprocessing is not None
            and preprocessing.get('as_uint8') is True
            and image.GetPixelID() != sitk.sitkUInt8
        ):
            image = sitk.RescaleIntensity(image)
            image = sitk.Cast(image, sitk.sitkUInt8)
    else:
        image = sitk_backend(image_filepath, preprocessing)

    return image


def sitk_image_info(image_filepath):
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_filepath)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    im_dims = np.asarray(reader.GetSize())
    # swap to YX
    im_dims[[0, 1]] = im_dims[[1, 0]]
    im_dtype = np.dtype(SITK_TO_NP_DTYPE.get(reader.GetPixelID()))
    is_vector = sitk.GetPixelIDValueAsString(reader.GetPixelID())

    if is_vector in "vector":
        im_dims = np.append(im_dims, 3)
    elif len(im_dims) == 3:
        im_dims = im_dims[[2, 0, 1]]
    else:
        im_dims = np.concatenate([[1], im_dims], axis=0)

    return im_dims, im_dtype


def get_im_info(image_filepath):
    if Path(image_filepath).suffix.lower() == ".czi":
        czi = CziFile(image_filepath)
        ch_dim_idx = czi.axes.index('C')
        y_dim_idx = czi.axes.index('Y')
        x_dim_idx = czi.axes.index('X')
        im_dims = np.array(czi.shape)[[ch_dim_idx, y_dim_idx, x_dim_idx]]
        im_dtype = czi.dtype
        reader = "czi"

    elif Path(image_filepath).suffix.lower() in TIFFFILE_EXTS:
        largest_series = tf_get_largest_series(image_filepath)
        zarr_im = zarr.open(
            imread(image_filepath, aszarr=True, series=largest_series)
        )
        zarr_im = zarr_get_base_pyr_layer(zarr_im)
        im_dims = np.squeeze(zarr_im.shape)
        if len(im_dims) == 2:
            im_dims = np.concatenate([[1], im_dims])
        im_dtype = zarr_im.dtype
        reader = "tifffile"

    else:
        im_dims, im_dtype = sitk_image_info(image_filepath)
        reader = "sitk"

    return im_dims, im_dtype, reader


def tf_zarr_read_single_ch(image_filepath, channel_idx, is_rgb):
    largest_series = tf_get_largest_series(image_filepath)
    zarr_im = zarr.open(
        imread(image_filepath, aszarr=True, series=largest_series)
    )
    zarr_im = zarr_get_base_pyr_layer(zarr_im)

    try:
        im = da.squeeze(da.from_zarr(zarr_im))
        if is_rgb is True:
            im = im[:, :, channel_idx].compute()
        elif len(im.shape) > 2:
            im = im[channel_idx, :, :].compute()
        else:
            im = im.compute()

    except ValueError:
        im = zarr_im
        if is_rgb is True:
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

    res_shape = xy_final_shape[::-1]
    res_shapes = [tuple(res_shape)]

    while all(res_shape > tile_size):
        res_shape = res_shape // 2
        res_shapes.append(tuple(res_shape))

    return res_shapes[:-1]


def add_ome_axes_single_plane(image_np):
    return image_np.reshape((1,) * (3) + image_np.shape)


def generate_channels(channel_names, channel_colors, im_dtype):
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
    if channel_names is None or n_ch != len(channel_names):
        channel_names = ["C{}".format(idx) for idx in range(n_ch)]
    return channel_names


def get_pyramid_info(y_size, x_size, n_ch, tile_size):
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


def get_final_tform_info(tform_dict):
    x_size, y_size = tform_dict.get(list(tform_dict.keys())[-1])[-1].get(
        "Size"
    )
    return int(y_size), int(x_size)


def image_to_zarr_store(zgrp, image, channel_idx, n_pyr_levels, pyr_levels):
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


def get_final_yx_from_tform(tform_reg_im):
    if tform_reg_im.tform_dict is not None:
        y_size, x_size = get_final_tform_info(tform_reg_im.tform_dict)
    else:
        y_size, x_size = (
            (tform_reg_im.im_dims[0], tform_reg_im.im_dims[1])
            if tform_reg_im.is_rgb
            else (tform_reg_im.im_dims[1], tform_reg_im.im_dims[2])
        )
    return y_size, x_size


def transform_to_ome_zarr(tform_reg_im, output_dir, tile_size):

    y_size, x_size = get_final_yx_from_tform(tform_reg_im)

    n_ch = (
        tform_reg_im.im_dims[2]
        if tform_reg_im.is_rgb
        else tform_reg_im.im_dims[0]
    )
    pyr_levels, pyr_shapes = get_pyramid_info(y_size, x_size, n_ch, tile_size)
    n_pyr_levels = len(pyr_levels)
    output_file_name = str(Path(output_dir) / tform_reg_im.image_name)
    if tform_reg_im.reader in ["tifffile", "czi"]:

        for channel_idx in range(n_ch):
            if tform_reg_im.reader == "tifffile":
                tform_reg_im.image = tf_zarr_read_single_ch(
                    tform_reg_im.image_filepath,
                    tform_reg_im.is_rgb,
                    channel_idx,
                )
                tform_reg_im.image = np.squeeze(tform_reg_im.image)
            elif tform_reg_im.reader == "czi":
                tform_reg_im.image = czi_read_single_ch(
                    tform_reg_im.image_filepath, channel_idx
                )
                tform_reg_im.image = np.squeeze(tform_reg_im.image)

            tform_reg_im.image = sitk.GetImageFromArray(
                tform_reg_im.image, isVector=False
            )
            tform_reg_im.image.SetSpacing(
                (tform_reg_im.image_res, tform_reg_im.image_res)
            )

            tform_reg_im.preprocess_reg_image_spatial(
                tform_reg_im.spatial_prepro, tform_reg_im.transforms
            )

            tform_reg_im.image = apply_transform_dict(
                tform_reg_im.image,
                tform_reg_im.image_res,
                tform_reg_im.tform_dict,
                is_shape_mask=False,
                writer="sitk",
            )
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


def transform_to_ome_tiff(tform_reg_im, output_dir, tile_size):
    y_size, x_size = get_final_yx_from_tform(tform_reg_im)

    n_ch = (
        tform_reg_im.im_dims[2]
        if tform_reg_im.is_rgb
        else tform_reg_im.im_dims[0]
    )
    pyr_levels, pyr_shapes = get_pyramid_info(y_size, x_size, n_ch, tile_size)
    n_pyr_levels = len(pyr_levels)
    output_file_name = str(Path(output_dir) / tform_reg_im.image_name)
    channel_names = format_channel_names(tform_reg_im.channel_names, n_ch)

    omexml = prepare_ome_xml_str(
        y_size,
        x_size,
        n_ch,
        tform_reg_im.im_dtype,
        tform_reg_im.is_rgb,
        PhysicalSizeX=tform_reg_im.image_res,
        PhysicalSizeY=tform_reg_im.image_res,
        PhysicalSizeXUnit="µm",
        PhysicalSizeYUnit="µm",
        Name=tform_reg_im.image_name,
        Channel=None if tform_reg_im.is_rgb else {"Name": channel_names},
    )

    if tform_reg_im.reader in ["tifffile", "czi"]:
        print(f"saving to {output_file_name}.ome.tiff")
        with TiffWriter(f"{output_file_name}.ome.tiff", bigtiff=True) as tif:
            for channel_idx in range(n_ch):
                if tform_reg_im.reader == "tifffile":
                    tform_reg_im.image = tf_zarr_read_single_ch(
                        tform_reg_im.image_filepath,
                        tform_reg_im.is_rgb,
                        channel_idx,
                    )
                    tform_reg_im.image = np.squeeze(tform_reg_im.image)
                elif tform_reg_im.reader == "czi":
                    tform_reg_im.image = czi_read_single_ch(
                        tform_reg_im.image_filepath, channel_idx
                    )
                    tform_reg_im.image = np.squeeze(tform_reg_im.image)

                tform_reg_im.image = sitk.GetImageFromArray(
                    tform_reg_im.image, isVector=False
                )
                tform_reg_im.image.SetSpacing(
                    (tform_reg_im.image_res, tform_reg_im.image_res)
                )

                tform_reg_im.preprocess_reg_image_spatial(
                    tform_reg_im.spatial_prepro, tform_reg_im.transforms
                )
                tform_reg_im.image = apply_transform_dict(
                    tform_reg_im.image,
                    tform_reg_im.image_res,
                    tform_reg_im.tform_dict,
                    is_shape_mask=False,
                    writer="sitk",
                )

                tform_reg_im.image = sitk.GetArrayFromImage(tform_reg_im.image)

                options = dict(
                    tile=(tile_size, tile_size),
                    compression="jpeg" if tform_reg_im.is_rgb else None,
                    photometric="rgb" if tform_reg_im.is_rgb else "minisblack",
                    metadata=None,
                )
                # write OME-XML to the ImageDescription tag of the first page
                description = omexml if channel_idx == 0 else None

                # write channel data
                tif.write(
                    tform_reg_im.image,
                    subifds=n_pyr_levels - 1,
                    description=description,
                    **options,
                )

                print(
                    f"channel {channel_idx} shape: {tform_reg_im.image.shape}"
                )

                for pyr_idx in range(1, n_pyr_levels):
                    resize_shape = (
                        pyr_levels[pyr_idx][0],
                        pyr_levels[pyr_idx][1],
                    )
                    tform_reg_im.image = cv2.resize(
                        tform_reg_im.image, resize_shape, cv2.INTER_LINEAR
                    )
                    print(
                        f"pyr {pyr_idx} : channel {channel_idx} shape: {tform_reg_im.image.shape}"
                    )

                    tif.write(tform_reg_im.image, **options, subfiletype=1)
    return f"{output_file_name}.ome.tiff"


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
        'as_uint8': False,
        'downsample': 1,
        'max_int_proj': sitk_max_int_proj,
        'inv_int': sitk_inv_int,
    }
    return STD_PREPRO
