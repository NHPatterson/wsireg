from pathlib import Path
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import SimpleITK as sitk
from tifffile import create_output, TiffFile, xml2dict, TiffWriter, imread
from czifile import CziFile
import zarr
import dask.array as da

TIFFFILE_EXTS = [".scn", ".tif", ".tiff", ".ndpi"]


def tifffile_zarr_backend(image_filepath, largest_series, preprocessing):

    print("using zarr backend")
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)

    if isinstance(zarr_store, zarr.hierarchy.Group):
        zarr_im = zarr_store[0]
    elif isinstance(zarr_store, zarr.core.Array):
        zarr_im = zarr_store

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

    if isinstance(zarr_store, zarr.hierarchy.Group):
        dask_im = da.squeeze(da.from_zarr(zarr_store[0]))
    elif isinstance(zarr_store, zarr.core.Array):
        dask_im = da.squeeze(da.from_zarr(zarr_store))

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


def sitk_to_ometiff(
    sitk_image,
    output_name,
    image_name=None,
    channel_names=None,
    image_res=None,
):

    if image_res is None:
        image_res = (None, None)

    # assume multicomponent RGB/A images
    if (
        sitk_image.GetNumberOfComponentsPerPixel() >= 3
        and sitk_image.GetNumberOfComponentsPerPixel() < 5
    ):
        photometric_tag = "rgb"
        channel_names = None
        axes = "YXS"
    else:
        photometric_tag = "minisblack"
        if (
            channel_names is None
            or len(channel_names) != sitk_image.GetDepth()
        ):
            channel_names = [
                "Ch{}".format(str(idx).zfill(3))
                for idx in range(sitk_image.GetDepth())
            ]
        axes = "CYX"

    sitk_image = sitk.GetArrayFromImage(sitk_image)

    if image_name is None:
        image_name = "default"

    with TiffWriter(output_name) as tif:
        tif.save(
            sitk_image,
            compress=9,
            tile=(1024, 1024),
            photometric=photometric_tag,
            metadata={
                "axes": axes,
                "SignificantBits": sitk_image.dtype.itemsize * 8,
                "Image": {
                    "Name": image_name,
                    "Pixels": {
                        "PhysicalSizeX": image_res[0],
                        "PhysicalSizeY": image_res[1],
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeYUnit": "µm",
                        "Channel": {"Name": channel_names},
                    },
                },
            },
        )
