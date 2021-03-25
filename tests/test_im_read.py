import pytest
import os
from wsireg.utils.im_utils import read_image, std_prepro
import SimpleITK as sitk

# private data logic borrowed from https://github.com/cgohlke/tifffile/tests/test_tifffile.py
HERE = os.path.dirname(__file__)
PRIVATE_DIR = os.path.join(HERE, "private_data")

SKIP_PRIVATE = False
REASON = "private data"

if not os.path.exists(PRIVATE_DIR):
    SKIP_PRIVATE = True


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_as_uint8():
    image_fp = os.path.join(PRIVATE_DIR, "testczi_useczifile.czi")
    prepro = std_prepro()
    prepro.update({"as_uint8": True})
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 3
    assert image.GetPixelID() == sitk.sitkUInt8


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_as_uint8_ch_indices_1ch():
    image_fp = os.path.join(PRIVATE_DIR, "testczi_useczifile.czi")
    prepro = std_prepro()
    prepro.update({"as_uint8": True, "ch_indices": [0]})
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_czi_read_as_uint8_ch_indices_2ch():
    image_fp = os.path.join(PRIVATE_DIR, "testczi_useczifile.czi")
    prepro = std_prepro()
    prepro.update({"as_uint8": True, "ch_indices": [0, 1]})
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 3
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 2


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_imagescope_tif_gs():
    image_fp = os.path.join(PRIVATE_DIR, "testimagescope_tif.tif")
    prepro = std_prepro()
    prepro.update({"as_uint8": True})
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_mc_tif():
    image_fp = os.path.join(PRIVATE_DIR, "testim_4ch_16bit.tiff")
    prepro = std_prepro()
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 3
    assert image.GetPixelID() == sitk.sitkUInt16
    assert image.GetDepth() == 4


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_mc_tif_1ch():
    image_fp = os.path.join(PRIVATE_DIR, "testim_4ch_16bit.tiff")
    prepro = std_prepro()
    prepro.update({"as_uint8": True, "ch_indices": [0]})
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_mc_tif_2ch():
    image_fp = os.path.join(PRIVATE_DIR, "testim_4ch_16bit.tiff")
    prepro = std_prepro()
    prepro.update({"as_uint8": True, "ch_indices": [0, 1]})
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 3
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 2


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_rgb_tif_std_prepro():
    image_fp = os.path.join(PRIVATE_DIR, "testim_rgb_8bit.tif")
    prepro = std_prepro()
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_rgb_tif_no_prepro():
    image_fp = os.path.join(PRIVATE_DIR, "testim_rgb_8bit.tif")
    image = read_image(image_fp, preprocessing=None)
    assert image.GetNumberOfComponentsPerPixel() == 3
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkVectorUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_rgb_tif_no_prepro():
    image_fp = os.path.join(PRIVATE_DIR, "testim_rgb_8bit.tif")
    image = read_image(image_fp, preprocessing=None)
    assert image.GetNumberOfComponentsPerPixel() == 3
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkVectorUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_rgb_jpg_std_prepro():
    image_fp = os.path.join(PRIVATE_DIR, "testimagescope_jpg.jpg")
    prepro = std_prepro()
    image = read_image(image_fp, preprocessing=prepro)
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_rgb_jpg_std_prepro_f():
    image_fp = os.path.join(PRIVATE_DIR, "testimagescope_jpg.jpg")
    image = read_image(image_fp, preprocessing=None)
    assert image.GetNumberOfComponentsPerPixel() == 3
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkVectorUInt8
    assert image.GetDepth() == 0


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
def test_rgb_huron_std_prepro_f():
    image_fp = os.path.join(PRIVATE_DIR, "testhuron.tif")
    image = read_image(image_fp, preprocessing=None)
    assert image.GetNumberOfComponentsPerPixel() == 3
    assert len(image.GetSize()) == 2
    assert image.GetPixelID() == sitk.sitkVectorUInt8
    assert image.GetDepth() == 0
