import os

import numpy as np
import pytest

from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.reg_images.loader import reg_image_loader

# private data logic borrowed from https://github.com/cgohlke/tifffile/tests/test_tifffile.py
HERE = os.path.dirname(__file__)
PRIVATE_DIR = os.path.join(HERE, "private_data")

SKIP_PRIVATE = False
REASON = "private data"

if not os.path.exists(PRIVATE_DIR):
    SKIP_PRIVATE = True


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_rgb():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    ri = reg_image_loader(image_fp, 1)
    assert len(ri.im_dims) == 3
    assert ri.im_dims[2] == 3
    assert ri.im_dtype == np.uint8
    assert ri.is_rgb is True


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_rgb_default_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    ri = reg_image_loader(image_fp, 1)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_rgb_bf_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    preprocessing = {"image_type": "BF"}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_rgb_bf_preprocess_default():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    preprocessing = ImagePreproParams()
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    ri = reg_image_loader(image_fp, 1)
    assert len(ri.im_dims) == 3
    assert ri.im_dims[0] == 4
    assert ri.im_dims[2] > 3
    assert ri.im_dtype == np.uint16
    assert ri.is_rgb is False


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_default_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    ri = reg_image_loader(image_fp, 1)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert ri.reg_image.GetPixelID() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_fl_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    preprocessing = {"image_type": "FL", "as_uint8": True}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert ri.reg_image.GetPixelID() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_std_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    preprocessing = ImagePreproParams()
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert ri.reg_image.GetPixelID() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_selectch_preprocess_list():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    preprocessing = {"ch_indices": [0]}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert ri.reg_image.GetPixelID() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_selectch_preprocess_int():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    preprocessing = {"ch_indices": 0}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert ri.reg_image.GetPixelID() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_read_channels():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.czi")
    ri = reg_image_loader(image_fp, 1)
    ch0 = ri.read_single_channel(0)
    ch1 = ri.read_single_channel(1)
    ch2 = ri.read_single_channel(2)
    ch3 = ri.read_single_channel(3)

    assert np.squeeze(ch0).shape == ri.im_dims[1:]
    assert np.squeeze(ch1).shape == ri.im_dims[1:]
    assert np.squeeze(ch2).shape == ri.im_dims[1:]
    assert np.squeeze(ch3).shape == ri.im_dims[1:]
    assert np.ndim(ch0) == 6
    assert np.ndim(ch1) == 6
    assert np.ndim(ch2) == 6
    assert np.ndim(ch3) == 6
    assert np.mean(ch1) > 0
    assert np.mean(ch2) > 0
    assert np.mean(ch3) > 0
    assert np.array_equal(ch0, ch1) is False
    assert np.array_equal(ch0, ch2) is False
    assert np.array_equal(ch0, ch3) is False
    assert np.array_equal(ch1, ch2) is False
    assert np.array_equal(ch1, ch3) is False
    assert np.array_equal(ch2, ch3) is False
    assert ch0.dtype == np.uint16
    assert ch1.dtype == np.uint16
    assert ch2.dtype == np.uint16
    assert ch3.dtype == np.uint16


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_rgb_read_channels():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.czi")
    ri = reg_image_loader(image_fp, 1)
    ch0 = ri.read_single_channel(0)
    ch1 = ri.read_single_channel(1)
    ch2 = ri.read_single_channel(2)

    assert np.squeeze(ch0).shape == ri.im_dims[:2]
    assert np.squeeze(ch1).shape == ri.im_dims[:2]
    assert np.squeeze(ch2).shape == ri.im_dims[:2]
    assert np.ndim(ch0) == 6
    assert np.ndim(ch1) == 6
    assert np.ndim(ch2) == 6
    assert np.array_equal(ch0, ch1) is False
    assert np.array_equal(ch0, ch2) is False
    assert np.array_equal(ch1, ch2) is False
    assert ch0.dtype == np.uint8
    assert ch1.dtype == np.uint8
    assert ch2.dtype == np.uint8


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_scn_read_rgb():
    image_fp = os.path.join(PRIVATE_DIR, "scn_rgb.scn")
    ri = reg_image_loader(image_fp, 1)
    assert len(ri.im_dims) == 3
    assert ri.im_dims[2] == 3
    assert ri.im_dtype == np.uint8
    assert ri.is_rgb is True


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_scn_read_rgb_default_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "scn_rgb.scn")
    ri = reg_image_loader(image_fp, 1)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_scn_read_rgb_bf_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "scn_rgb.scn")
    preprocessing = {"image_type": "BF"}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_huron_read_rgb():
    image_fp = os.path.join(PRIVATE_DIR, "huron_rgb.tif")
    ri = reg_image_loader(image_fp, 1)
    assert len(ri.im_dims) == 3
    assert ri.im_dims[2] == 3
    assert ri.im_dtype == np.uint8
    assert ri.is_rgb is True


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_huron_read_rgb_default_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "huron_rgb.tif")
    ri = reg_image_loader(image_fp, 1)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_huron_read_rgb_bf_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "huron_rgb.tif")
    preprocessing = {"image_type": "BF"}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_huron_read_rgb_bf_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "huron_rgb.tif")
    preprocessing = {"image_type": "BF"}
    ri = reg_image_loader(image_fp, 1, preprocessing=preprocessing)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_ometiff_read_rgb():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.ome.tiff")
    ri = reg_image_loader(image_fp, 1)
    assert len(ri.im_dims) == 3
    assert ri.im_dtype == np.uint8
    assert ri.is_rgb is True


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_ometiff_read_rgb_default_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.ome.tiff")
    ri = reg_image_loader(image_fp, 1)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_ometiff_read_mc():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.ome.tiff")
    ri = reg_image_loader(image_fp, 1)

    assert len(ri.im_dims) == 3
    assert ri.im_dims[0] == 4
    assert ri.im_dims[2] > 3
    assert ri.im_dtype == np.uint16
    assert ri.is_rgb is False


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_ometiff_read_mc_default_preprocess():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.ome.tiff")
    ri = reg_image_loader(image_fp, 1)
    ri.read_reg_image()
    assert ri.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_mc_read_channels():
    image_fp = os.path.join(PRIVATE_DIR, "czi_4ch_16bit.ome.tiff")
    ri = reg_image_loader(image_fp, 1)
    ch0 = ri.read_single_channel(0)
    ch1 = ri.read_single_channel(1)
    ch2 = ri.read_single_channel(2)
    ch3 = ri.read_single_channel(3)

    assert np.squeeze(ch0).shape == ri.im_dims[1:]
    assert np.squeeze(ch1).shape == ri.im_dims[1:]
    assert np.squeeze(ch2).shape == ri.im_dims[1:]
    assert np.squeeze(ch3).shape == ri.im_dims[1:]
    assert np.ndim(ch0) == 2
    assert np.ndim(ch1) == 2
    assert np.ndim(ch2) == 2
    assert np.ndim(ch3) == 2
    assert np.array_equal(ch0, ch1) is False
    assert np.array_equal(ch0, ch2) is False
    assert np.array_equal(ch0, ch3) is False
    assert np.array_equal(ch1, ch2) is False
    assert np.array_equal(ch1, ch3) is False
    assert np.array_equal(ch2, ch3) is False
    assert ch0.dtype == np.uint16
    assert ch1.dtype == np.uint16
    assert ch2.dtype == np.uint16
    assert ch3.dtype == np.uint16


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_rgb_read_channels():
    image_fp = os.path.join(PRIVATE_DIR, "czi_rgb.ome.tiff")
    ri = reg_image_loader(image_fp, 1)
    ch0 = ri.read_single_channel(0)
    ch1 = ri.read_single_channel(1)
    ch2 = ri.read_single_channel(2)

    assert np.squeeze(ch0).shape == ri.im_dims[:2]
    assert np.squeeze(ch1).shape == ri.im_dims[:2]
    assert np.squeeze(ch2).shape == ri.im_dims[:2]
    assert np.ndim(ch0) == 2
    assert np.ndim(ch1) == 2
    assert np.ndim(ch2) == 2
    assert np.array_equal(ch0, ch1) is False
    assert np.array_equal(ch0, ch2) is False
    assert np.array_equal(ch1, ch2) is False
    assert ch0.dtype == np.uint8
    assert ch1.dtype == np.uint8
    assert ch2.dtype == np.uint8
