import dask.array as da
import numpy as np
import pytest
import zarr
from tifffile import TiffWriter, imread, imwrite


@pytest.fixture
def im_gry_np():
    return np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)


@pytest.fixture
def mask_np():
    mask_im = np.zeros((2048, 2048), dtype=np.uint8)
    mask_im[256:1792, 256:1792] = 255
    return mask_im


@pytest.fixture
def im_mch_np():
    return np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint16)


@pytest.fixture
def im_rgb_np():
    return np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)


@pytest.fixture
def dask_im_gry_np():
    return da.from_array(
        np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)
    )


@pytest.fixture
def dask_im_mch_np():
    return da.from_array(
        np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint16)
    )


@pytest.fixture
def dask_im_rgb_np():
    return da.from_array(
        np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    )


@pytest.fixture
def zarr_im_gry_np():
    return zarr.array(np.random.randint(0, 255, (2048, 2048), dtype=np.uint16))


@pytest.fixture
def zarr_im_mch_np():
    return zarr.array(
        np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint16)
    )


@pytest.fixture
def zarr_im_rgb_np():
    return zarr.array(
        np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    )


@pytest.fixture
def disk_im_mch(tmpdir_factory, im_mch_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_mch.tiff")
    imwrite(out_im, im_mch_np, tile=(256, 256))
    return out_im


@pytest.fixture
def disk_im_mch_notile(tmpdir_factory, im_mch_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_mch_nt.tiff")
    imwrite(out_im, im_mch_np, photometric="MINISBLACK", tile=(256, 256))
    return out_im


@pytest.fixture
def disk_im_rgb(tmpdir_factory, im_rgb_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_rgb.tiff")
    imwrite(out_im, im_rgb_np, tile=(256, 256))
    return out_im


@pytest.fixture
def disk_im_gry(tmpdir_factory, im_gry_np):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_gry.tiff")
    imwrite(out_im, im_gry_np, tile=(256, 256))
    return out_im


@pytest.fixture
def disk_im_mch_pyr(tmpdir_factory):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_mch_pyr.tiff")
    subifds = 2
    full_im = np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint16)
    with TiffWriter(out_im) as tif:
        for ch in range(full_im.shape[0]):
            options = dict(
                tile=(256, 256),
                compression="deflate",
                photometric="minisblack",
                metadata=None,
            )
            tif.write(
                full_im[ch, :, :],
                subifds=subifds,
                **options,
            )

            for pyr_idx in range(subifds):
                if pyr_idx == 0:
                    subresimage = full_im[ch, ::2, ::2]
                else:
                    subresimage = subresimage[::2, ::2]

                tif.write(subresimage, **options, subfiletype=1)
    return out_im


@pytest.fixture
def disk_im_rgb_pyr(tmpdir_factory):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_rgb_pyr.tiff")
    subifds = 2
    full_im = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    with TiffWriter(out_im) as tif:
        options = dict(
            tile=(128, 128),
            compression="deflate",
            photometric="rgb",
            metadata=None,
        )
        tif.write(
            full_im,
            subifds=subifds,
            **options,
        )

        for pyr_idx in range(subifds):
            if pyr_idx == 0:
                subresimage = full_im[::2, ::2, :]
            else:
                subresimage = subresimage[::2, ::2, :]

            tif.write(subresimage, **options, subfiletype=1)

    return out_im


@pytest.fixture
def disk_im_gry_pyr(tmpdir_factory):
    out_im = tmpdir_factory.mktemp("image").join("image_fp_gry_pyr.tiff")
    subifds = 2
    full_im = np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)
    with TiffWriter(out_im) as tif:
        options = dict(
            tile=(256, 256),
            compression="deflate",
            photometric="minisblack",
            metadata=None,
        )
        tif.write(
            full_im,
            subifds=subifds,
            **options,
        )

        for pyr_idx in range(subifds):
            if pyr_idx == 0:
                subresimage = full_im[::2, ::2]
            else:
                subresimage = subresimage[::2, ::2]

            tif.write(subresimage, **options, subfiletype=1)
    return out_im
