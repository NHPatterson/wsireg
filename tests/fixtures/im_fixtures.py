import pytest
import numpy as np
from tifffile import imwrite


@pytest.fixture
def im_gry_np():
    return np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)

@pytest.fixture
def mask_np():
    mask_im = np.zeros((2048, 2048), dtype=np.uint8)
    mask_im[256:1792,256:1792] = 255
    return mask_im

@pytest.fixture
def im_mch_np():
    return np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint16)

@pytest.fixture
def im_rgb_np():
    return np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)


@pytest.fixture
def disk_im_mch(tmpdir_factory, im_mch_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_mch.tiff")
    imwrite(out_im, im_mch_np)
    return out_im


@pytest.fixture
def disk_im_rgb(tmpdir_factory, im_rgb_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_rgb.tiff")
    imwrite(out_im, im_rgb_np)
    return out_im

@pytest.fixture
def disk_im_gry(tmpdir_factory, im_gry_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_gry.tiff")
    imwrite(out_im, im_gry_np)
    return out_im
