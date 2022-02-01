import os

import dask.array as da
import numpy as np
import pytest
import zarr
from tifffile import imread

from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.utils.im_utils import (
    CziRegImageReader,
    czi_tile_grayscale,
    ensure_dask_array,
    get_sitk_image_info,
    get_tifffile_info,
    grayscale,
    guess_rgb,
    read_preprocess_array,
    sitk_backend,
    tf_get_largest_series,
    tifffile_dask_backend,
    tifffile_zarr_backend,
    zarr_get_base_pyr_layer,
)

# private data logic borrowed from https://github.com/cgohlke/tifffile/tests/test_tifffile.py
HERE = os.path.dirname(__file__)
PRIVATE_DIR = os.path.join(HERE, "private_data")

SKIP_PRIVATE = False
REASON = "private data"

if not os.path.exists(PRIVATE_DIR):
    SKIP_PRIVATE = True


@pytest.mark.usefixtures(
    "disk_im_gry_pyr",
    "disk_im_mch_pyr",
    "disk_im_rgb_pyr",
    "disk_im_gry",
    "disk_im_mch",
    "disk_im_rgb",
)
def test_zarr_get_base_pyr_layer(
    disk_im_gry_pyr,
    disk_im_mch_pyr,
    disk_im_rgb_pyr,
    disk_im_gry,
    disk_im_mch,
    disk_im_rgb,
):
    zarr_store = zarr.open(imread(disk_im_gry_pyr, aszarr=True))
    gry_zarray_from_pyr = zarr_get_base_pyr_layer(zarr_store)

    zarr_store = zarr.open(imread(disk_im_mch_pyr, aszarr=True))
    mch_zarray_from_pyr = zarr_get_base_pyr_layer(zarr_store)

    zarr_store = zarr.open(imread(disk_im_rgb_pyr, aszarr=True))
    rgb_zarray_from_pyr = zarr_get_base_pyr_layer(zarr_store)

    zarr_store = zarr.open(imread(disk_im_gry, aszarr=True))
    gry_zarray_from_flat = zarr_get_base_pyr_layer(zarr_store)

    zarr_store = zarr.open(imread(disk_im_mch, aszarr=True))
    mch_zarray_from_flat = zarr_get_base_pyr_layer(zarr_store)

    zarr_store = zarr.open(imread(disk_im_rgb, aszarr=True))
    rgb_zarray_from_flat = zarr_get_base_pyr_layer(zarr_store)

    assert gry_zarray_from_pyr.shape == (2048, 2048)
    assert mch_zarray_from_pyr.shape == (3, 2048, 2048)
    assert rgb_zarray_from_pyr.shape == (2048, 2048, 3)
    assert gry_zarray_from_flat.shape == (2048, 2048)
    assert mch_zarray_from_flat.shape == (3, 2048, 2048)
    assert rgb_zarray_from_flat.shape == (2048, 2048, 3)


def test_ensure_dask_array():
    np_arr = np.zeros((128, 128, 3), dtype=np.uint8)
    da_arr = da.zeros((128, 128, 3), dtype=np.uint8)
    za_arr = zarr.zeros((128, 128, 3), dtype=np.uint8)

    np_out = ensure_dask_array(np_arr)
    da_out = ensure_dask_array(da_arr)
    za_out = ensure_dask_array(za_arr)

    assert isinstance(np_out, da.Array)
    assert isinstance(da_out, da.Array)
    assert isinstance(za_out, da.Array)


def test_read_preprocess_array():
    da_arr = ensure_dask_array(np.zeros((128, 128, 3), dtype=np.uint8))
    mc_arr = ensure_dask_array(np.zeros((3, 128, 128), dtype=np.uint8))
    gr_arr = ensure_dask_array(np.zeros((128, 128), dtype=np.uint8))

    std_rgb = read_preprocess_array(
        da_arr,
        preprocessing=ImagePreproParams(image_type="BF"),
        force_rgb=None,
    )
    nonstd_rgb = read_preprocess_array(
        mc_arr,
        preprocessing=ImagePreproParams(image_type="BF"),
        force_rgb=True,
    )

    all_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(image_type="FL")
    )

    ch0_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(ch_indices=[0])
    )
    ch01_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(ch_indices=[0, 1])
    )
    ch12_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(ch_indices=[1, 2])
    )
    ch02_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(ch_indices=[0, 2])
    )
    ch1_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(ch_indices=[1])
    )
    ch2_ch_mc = read_preprocess_array(
        mc_arr, preprocessing=ImagePreproParams(ch_indices=[2])
    )

    std_gr = read_preprocess_array(gr_arr, ImagePreproParams(image_type="FL"))
    chsel0_gr = read_preprocess_array(
        gr_arr, preprocessing=ImagePreproParams(ch_indices=[0])
    )
    chsel01_gr = read_preprocess_array(
        gr_arr, preprocessing=ImagePreproParams(ch_indices=[0, 1])
    )

    assert std_rgb.GetNumberOfComponentsPerPixel() == 1
    assert nonstd_rgb.GetNumberOfComponentsPerPixel() == 1
    assert all_ch_mc.GetDepth() == 3
    assert ch0_ch_mc.GetDepth() == 0
    assert ch01_ch_mc.GetDepth() == 2
    assert ch12_ch_mc.GetDepth() == 2
    assert ch02_ch_mc.GetDepth() == 2
    assert ch1_ch_mc.GetDepth() == 0
    assert ch2_ch_mc.GetDepth() == 0
    assert std_gr.GetDepth() == 0
    assert chsel0_gr.GetDepth() == 0
    assert chsel01_gr.GetDepth() == 0
    assert std_rgb.GetSize() == (128, 128)
    assert nonstd_rgb.GetSize() == (128, 128)
    assert all_ch_mc.GetSize() == (128, 128, 3)
    assert ch0_ch_mc.GetSize() == (128, 128)
    assert ch01_ch_mc.GetSize() == (128, 128, 2)
    assert ch12_ch_mc.GetSize() == (128, 128, 2)
    assert ch02_ch_mc.GetSize() == (128, 128, 2)
    assert ch1_ch_mc.GetSize() == (128, 128)
    assert ch2_ch_mc.GetSize() == (128, 128)
    assert std_gr.GetSize() == (128, 128)
    assert chsel0_gr.GetSize() == (128, 128)
    assert chsel01_gr.GetSize() == (128, 128)


@pytest.mark.usefixtures(
    "disk_im_gry_pyr",
    "disk_im_mch_pyr",
    "disk_im_rgb_pyr",
    "disk_im_gry",
    "disk_im_mch",
    "disk_im_rgb",
)
def test_tifffile_zarr_backend(
    disk_im_gry_pyr,
    disk_im_mch_pyr,
    disk_im_rgb_pyr,
    disk_im_gry,
    disk_im_mch,
    disk_im_rgb,
):
    im_gry_pyr = tifffile_zarr_backend(
        disk_im_gry_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_mch_pyr = tifffile_zarr_backend(
        disk_im_mch_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_rgb_pyr = tifffile_zarr_backend(
        disk_im_rgb_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="BF"),
    )
    im_gry = tifffile_zarr_backend(
        disk_im_gry,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_mch = tifffile_zarr_backend(
        disk_im_mch,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_rgb = tifffile_zarr_backend(
        disk_im_rgb,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="BF"),
    )

    assert im_gry_pyr.GetSize() == (2048, 2048)
    assert im_mch_pyr.GetSize() == (2048, 2048, 3)
    assert im_rgb_pyr.GetSize() == (2048, 2048)
    assert im_gry.GetSize() == (2048, 2048)
    assert im_mch.GetSize() == (2048, 2048, 3)
    assert im_rgb.GetSize() == (2048, 2048)


@pytest.mark.usefixtures(
    "disk_im_gry_pyr",
    "disk_im_mch_pyr",
    "disk_im_rgb_pyr",
    "disk_im_gry",
    "disk_im_mch",
    "disk_im_rgb",
)
def test_tifffile_dask_backend(
    disk_im_gry_pyr,
    disk_im_mch_pyr,
    disk_im_rgb_pyr,
    disk_im_gry,
    disk_im_mch,
    disk_im_rgb,
):
    im_gry_pyr = tifffile_dask_backend(
        disk_im_gry_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_mch_pyr = tifffile_dask_backend(
        disk_im_mch_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_rgb_pyr = tifffile_dask_backend(
        disk_im_rgb_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="BF"),
    )
    im_gry = tifffile_dask_backend(
        disk_im_gry,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_mch = tifffile_dask_backend(
        disk_im_mch,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_rgb = tifffile_dask_backend(
        disk_im_rgb,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="BF"),
    )

    assert im_gry_pyr.GetSize() == (2048, 2048)
    assert im_mch_pyr.GetSize() == (2048, 2048, 3)
    assert im_rgb_pyr.GetSize() == (2048, 2048)
    assert im_gry.GetSize() == (2048, 2048)
    assert im_mch.GetSize() == (2048, 2048, 3)
    assert im_rgb.GetSize() == (2048, 2048)


@pytest.mark.usefixtures(
    "disk_im_gry_pyr",
    "disk_im_mch_pyr",
    "disk_im_rgb_pyr",
    "disk_im_gry",
    "disk_im_mch",
    "disk_im_rgb",
)
def test_tifffile_dask_backend(
    disk_im_gry_pyr,
    disk_im_mch_pyr,
    disk_im_rgb_pyr,
    disk_im_gry,
    disk_im_mch,
    disk_im_rgb,
):
    im_gry_pyr = tifffile_dask_backend(
        disk_im_gry_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_mch_pyr = tifffile_dask_backend(
        disk_im_mch_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_rgb_pyr = tifffile_dask_backend(
        disk_im_rgb_pyr,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="BF"),
    )
    im_gry = tifffile_dask_backend(
        disk_im_gry,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_mch = tifffile_dask_backend(
        disk_im_mch,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="FL"),
    )
    im_rgb = tifffile_dask_backend(
        disk_im_rgb,
        largest_series=0,
        preprocessing=ImagePreproParams(image_type="BF"),
    )

    assert im_gry_pyr.GetSize() == (2048, 2048)
    assert im_mch_pyr.GetSize() == (2048, 2048, 3)
    assert im_rgb_pyr.GetSize() == (2048, 2048)
    assert im_gry.GetSize() == (2048, 2048)
    assert im_mch.GetSize() == (2048, 2048, 3)
    assert im_rgb.GetSize() == (2048, 2048)


@pytest.mark.usefixtures(
    "disk_im_gry",
    "disk_im_mch_notile",
    "disk_im_rgb",
)
def test_sitk_backend(
    disk_im_gry,
    disk_im_mch_notile,
    disk_im_rgb,
):

    im_gry = sitk_backend(
        str(disk_im_gry), preprocessing=ImagePreproParams(image_type="FL")
    )
    # im_mch = sitk_backend(str(disk_im_mch_notile), preprocessing={"image_type":"FL"})
    im_rgb = sitk_backend(
        str(disk_im_rgb), preprocessing=ImagePreproParams(image_type="BF")
    )

    assert im_gry.GetSize() == (2048, 2048)
    # assert im_mch.GetSize() == (2048,2048,3)
    assert im_rgb.GetSize() == (2048, 2048)


def test_guess_rgb():
    imshape_gs = (2048, 2048)
    imshape_mch = (3, 2048, 2048)
    imshape_rgb = (2048, 2048, 3)
    imshape_rgba = (2048, 2048, 4)

    assert guess_rgb(imshape_gs) is False
    assert guess_rgb(imshape_mch) is False
    assert guess_rgb(imshape_rgb) is True
    assert guess_rgb(imshape_rgba) is True


def test_greyscale():
    rgb_image_il = np.ones((2048, 2048, 3), dtype=np.uint8)
    rgb_image_noil = np.ones((3, 2048, 2048), dtype=np.uint8)

    da_rgb_image_il = da.ones(
        (2048, 2048, 3), dtype=np.uint8, chunks=(512, 512, 3)
    )
    da_rgb_image_noil = da.ones(
        (3, 2048, 2048), dtype=np.uint8, chunks=(512, 512, 3)
    )

    gs_il = grayscale(rgb_image_il, is_interleaved=True)
    gs_noil = grayscale(rgb_image_noil, is_interleaved=False)

    da_gs_il = grayscale(da_rgb_image_il, is_interleaved=True)
    da_gs_noil = grayscale(da_rgb_image_noil, is_interleaved=False)

    assert gs_il.shape == (2048, 2048)
    assert gs_noil.shape == (2048, 2048)
    assert np.array_equal(gs_il, gs_noil)
    assert da_gs_il.shape == (2048, 2048)
    assert da_gs_noil.shape == (2048, 2048)
    assert np.array_equal(da_gs_il.compute(), da_gs_noil.compute())


# PRIVATE_DIR = "tests/private_data"
@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_tile_greyscale():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    czi = CziRegImageReader(image_fp)
    fsbd = czi.filtered_subblock_directory[5]
    subblock = fsbd.data_segment()
    tile = subblock.data(resize=False, order=0)
    tile_gs = czi_tile_grayscale(tile)

    assert tile.shape[:5] == tile_gs.shape[:5]
    assert tile_gs.shape[-1] == 1
    assert tile_gs.dtype == np.uint8


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_CziRegImageReader_rgb():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    czi = CziRegImageReader(image_fp)
    gs_out = czi.sub_asarray_rgb(greyscale=True)
    rgb_out = czi.sub_asarray_rgb(greyscale=False)

    assert len(np.squeeze(gs_out).shape) == 2
    assert len(np.squeeze(rgb_out).shape) == 3


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_CziRegImageReader_mc():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    czi = CziRegImageReader(image_fp)
    mc_out = czi.asarray(max_workers=1)
    ch0_out = czi.sub_asarray(channel_idx=[0], max_workers=1)
    ch1_out = czi.sub_asarray(channel_idx=[1], max_workers=1)
    ch2_out = czi.sub_asarray(channel_idx=[2], max_workers=1)
    ch3_out = czi.sub_asarray(channel_idx=[3], max_workers=1)
    ch02_out = czi.sub_asarray(channel_idx=[0, 2], max_workers=1)
    ch13_out = czi.sub_asarray(channel_idx=[1, 3], max_workers=1)
    ch03_out = czi.sub_asarray(channel_idx=[0, 3], max_workers=1)
    ch12_out = czi.sub_asarray(channel_idx=[1, 2], max_workers=1)
    ch23_out = czi.sub_asarray(channel_idx=[2, 3], max_workers=1)

    assert np.squeeze(mc_out).shape == (4, 4305, 4194)
    assert np.squeeze(ch0_out).shape == (4305, 4194)
    assert np.squeeze(ch1_out).shape == (4305, 4194)
    assert np.squeeze(ch1_out).shape == (4305, 4194)
    assert np.squeeze(ch2_out).shape == (4305, 4194)
    assert np.squeeze(ch3_out).shape == (4305, 4194)
    assert np.squeeze(ch02_out).shape == (2, 4305, 4194)
    assert np.array_equal(
        np.squeeze(ch02_out), np.squeeze(mc_out)[[0, 2], :, :]
    )
    assert np.array_equal(
        np.squeeze(ch13_out), np.squeeze(mc_out)[[1, 3], :, :]
    )
    assert np.array_equal(
        np.squeeze(ch03_out), np.squeeze(mc_out)[[0, 3], :, :]
    )
    assert np.array_equal(
        np.squeeze(ch12_out), np.squeeze(mc_out)[[1, 2], :, :]
    )
    assert np.array_equal(
        np.squeeze(ch23_out), np.squeeze(mc_out)[[2, 3], :, :]
    )
    assert np.array_equal(np.squeeze(mc_out)[0, :, :], np.squeeze(ch0_out))
    assert np.array_equal(np.squeeze(mc_out)[1, :, :], np.squeeze(ch1_out))
    assert np.array_equal(np.squeeze(mc_out)[2, :, :], np.squeeze(ch2_out))
    assert np.array_equal(np.squeeze(mc_out)[3, :, :], np.squeeze(ch3_out))


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_tf_get_largest_series():
    image_fp_czi_mc = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.ome.tiff")
    image_fp_czi_rgb = os.path.join(PRIVATE_DIR, "czi_rgb.ome.tiff")
    image_fp_bigtiff = os.path.join(PRIVATE_DIR, "huron_rgb.tif")
    image_fp_scn = os.path.join(PRIVATE_DIR, "scn_rgb.scn")
    assert tf_get_largest_series(image_fp_czi_mc) == 0
    assert tf_get_largest_series(image_fp_czi_rgb) == 0
    assert tf_get_largest_series(image_fp_bigtiff) == 0
    assert tf_get_largest_series(image_fp_scn) == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_get_sitk_image_info():
    image_fp_jpgrgb = os.path.join(PRIVATE_DIR, "testjpegrgb.jpg")
    image_fp_pngrgb = os.path.join(PRIVATE_DIR, "testpngrgb.png")

    assert np.array_equal(
        get_sitk_image_info(image_fp_jpgrgb)[0], [450, 750, 3]
    )
    assert get_sitk_image_info(image_fp_jpgrgb)[1] == np.uint8
    assert np.array_equal(
        get_sitk_image_info(image_fp_pngrgb)[0], [450, 750, 3]
    )
    assert get_sitk_image_info(image_fp_pngrgb)[1] == np.uint8


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_get_tifffile_info_private():
    image_fp_bigtiff = os.path.join(PRIVATE_DIR, "huron_rgb.tif")
    image_fp_scn = os.path.join(PRIVATE_DIR, "scn_rgb.scn")

    assert np.array_equal(
        get_tifffile_info(image_fp_bigtiff)[0], [7662, 15778, 3]
    )
    assert get_tifffile_info(image_fp_bigtiff)[1] == np.uint8
    assert np.array_equal(
        get_tifffile_info(image_fp_scn)[0], [11776, 18528, 3]
    )
    assert get_tifffile_info(image_fp_scn)[1] == np.uint8


@pytest.mark.usefixtures(
    "disk_im_gry_pyr",
    "disk_im_mch_pyr",
    "disk_im_rgb_pyr",
)
def test_get_tifffile_info_public(
    disk_im_gry_pyr,
    disk_im_mch_pyr,
    disk_im_rgb_pyr,
):

    assert np.array_equal(
        get_tifffile_info(disk_im_rgb_pyr)[0], [2048, 2048, 3]
    )
    assert get_tifffile_info(disk_im_rgb_pyr)[1] == np.uint8

    assert np.array_equal(
        get_tifffile_info(disk_im_mch_pyr)[0], [3, 2048, 2048]
    )
    assert get_tifffile_info(disk_im_mch_pyr)[1] == np.uint16

    assert np.array_equal(
        get_tifffile_info(disk_im_gry_pyr)[0], [1, 2048, 2048]
    )
    assert get_tifffile_info(disk_im_gry_pyr)[1] == np.uint16
