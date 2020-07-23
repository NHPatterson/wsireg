from pathlib import Path
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import SimpleITK as sitk
from tifffile import create_output, TiffFile
from czifile import CziFile


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
    fp_ext = Path(image_filepath).suffix

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
    elif fp_ext == '.scn':

        image = TiffFile(image_filepath).series[3].asarray()
        image = sitk.GetImageFromArray(image, isVector=True)
        if preprocessing is not None:
            image = sitk_vect_to_gs(image)

    else:
        image = sitk.ReadImage(str(image_filepath))
        if (
            preprocessing is not None
            and image.GetNumberOfComponentsPerPixel() > 2
        ):
            image = sitk_vect_to_gs(image)

        if preprocessing is not None:
            if preprocessing['ch_indices'] is not None:
                image = image[:, :, preprocessing['ch_indices']]

            if (
                preprocessing['as_uint8'] is True
                and image.GetPixelID() is not sitk.sitkUInt8
            ):
                image = sitk.RescaleIntensity(image)
                image = sitk.Cast(image, sitk.sitkUInt8)

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
        SimpleITK image

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
        'as_uint8': None,
        'downsample': 1,
        'max_int_proj': sitk_max_int_proj,
        'inv_int': sitk_inv_int,
        'contrast_enhance': contrast_enhance,
    }
    return STD_PREPRO
