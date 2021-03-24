import pytest
from wsireg.reg_image import RegImage, TransformRegImage
import SimpleITK as sitk
import itk
import numpy as np
from tifffile import imread
import zarr


@pytest.mark.usefixtures("disk_im_mch")
def test_RegImage_image_fp_mc_std_prepro(disk_im_mch):
    reg_image = RegImage(str(disk_im_mch), 0.65)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_rgb")
def test_RegImage_image_fp_rgb_std_prepro(disk_im_rgb):
    reg_image = RegImage(str(disk_im_rgb), 0.65)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_gry")
def test_RegImage_image_fp_gry_std_prepro(disk_im_gry):
    reg_image = RegImage(str(disk_im_gry), 0.65)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_mch")
def test_RegImage_image_fp_mch_no_prepro(disk_im_mch):
    reg_image = RegImage(str(disk_im_mch), 0.65, prepro_dict=None)
    assert reg_image.image.GetSize() == (2048, 2048, 3)
    assert reg_image.image.GetDepth() == 3


@pytest.mark.usefixtures("disk_im_rgb")
def test_RegImage_image_fp_rgb_no_prepro(disk_im_rgb):
    reg_image = RegImage(str(disk_im_rgb), 0.65, prepro_dict=None)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 3


@pytest.mark.usefixtures("disk_im_gry")
def test_RegImage_image_fp_gry_no_prepro(disk_im_gry):
    reg_image = RegImage(str(disk_im_gry), 0.65, prepro_dict=None)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0


@pytest.mark.usefixtures("im_mch_np")
def test_RegImage_image_np_mc_std_prepro(im_mch_np):
    reg_image = RegImage(im_mch_np, 0.65)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0


@pytest.mark.usefixtures("im_rgb_np")
def test_RegImage_image_np_rgb_std_prepro(im_rgb_np):
    reg_image = RegImage(im_rgb_np, 0.65)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("im_gry_np")
def test_RegImage_image_np_gry_std_prepro(im_gry_np):
    reg_image = RegImage(im_gry_np, 0.65)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("im_mch_np")
def test_RegImage_image_np_no_prepro(im_mch_np):
    reg_image = RegImage(im_mch_np, 0.65, prepro_dict=None)
    assert reg_image.image.GetSize() == (2048, 2048, 3)


@pytest.mark.usefixtures("im_rgb_np")
def test_RegImage_image_np_rgb_no_prepro(im_rgb_np):
    reg_image = RegImage(im_rgb_np, 0.65, prepro_dict=None)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 3


@pytest.mark.usefixtures("im_gry_np")
def test_RegImage_image_np_gry_no_prepro(im_gry_np):
    reg_image = RegImage(im_gry_np, 0.65, prepro_dict=None)
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0


@pytest.mark.usefixtures("im_mch_np")
def test_RegImage_image_np_mc_std_prepro_rot(im_mch_np):
    reg_image = RegImage(im_mch_np, 0.65, prepro_dict={"rot_cc": 90})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_RegImage_image_np_rgb_std_prepro_rot(im_rgb_np):
    reg_image = RegImage(im_rgb_np, 0.65, prepro_dict={"rot_cc": 90})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_RegImage_image_np_gry_std_prepro_rot(im_gry_np):
    reg_image = RegImage(im_gry_np, 0.65, prepro_dict={"rot_cc": 90})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_RegImage_image_np_mc_std_prepro_fliph(im_mch_np):
    reg_image = RegImage(im_mch_np, 0.65, prepro_dict={"flip": "h"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_RegImage_image_np_rgb_std_prepro_fliph(im_rgb_np):
    reg_image = RegImage(im_rgb_np, 0.65, prepro_dict={"flip": "h"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_RegImage_image_np_gry_std_prepro_fliph(im_gry_np):
    reg_image = RegImage(im_gry_np, 0.65, prepro_dict={"flip": "h"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_RegImage_image_np_mc_std_prepro_flipv(im_mch_np):
    reg_image = RegImage(im_mch_np, 0.65, prepro_dict={"flip": "v"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_RegImage_image_np_rgb_std_prepro_flipv(im_rgb_np):
    reg_image = RegImage(im_rgb_np, 0.65, prepro_dict={"flip": "v"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_RegImage_image_np_gry_std_prepro_flipv(im_gry_np):
    reg_image = RegImage(im_gry_np, 0.65, prepro_dict={"flip": "v"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_RegImage_image_np_mc_std_prepro_rot_flipv(im_mch_np):
    reg_image = RegImage(
        im_mch_np, 0.65, prepro_dict={"rot_cc": 90, "flip": "v"}
    )
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 2
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_RegImage_image_np_gry_std_prepro_flipv(im_gry_np):
    reg_image = RegImage(im_gry_np, 0.65, prepro_dict={"flip": "v"})
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.transforms is not None
    assert len(reg_image.transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_mask(im_gry_np, mask_np):
    reg_image = RegImage(im_gry_np, 0.65, mask=mask_np)
    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_mask_rot(im_gry_np, mask_np):
    reg_image = RegImage(
        im_gry_np, 0.65, prepro_dict={"rot_cc": 90}, mask=mask_np
    )
    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_mask_rot_flip(im_gry_np, mask_np):
    reg_image = RegImage(
        im_gry_np, 0.65, prepro_dict={"rot_cc": 90, "flip": "h"}, mask=mask_np
    )
    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_mask_flip(im_gry_np, mask_np):
    reg_image = RegImage(
        im_gry_np, 0.65, prepro_dict={"flip": "v"}, mask=mask_np
    )
    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_downsampling(im_gry_np, mask_np):
    reg_image = RegImage(
        im_gry_np, 1, prepro_dict={"downsample": 2}, mask=mask_np
    )
    assert reg_image.image.GetSize() == (1024, 1024)
    assert reg_image.mask.GetSize() == (1024, 1024)
    assert reg_image.mask.GetSpacing() == (2, 2)
    assert reg_image.image.GetSpacing() == (2, 2)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_to_itk(im_gry_np, mask_np):
    reg_image = RegImage(im_gry_np, 0.65, mask=mask_np)
    reg_image.sitk_to_itk(cast_to_float32=True)
    assert isinstance(reg_image.image, itk.Image) is True
    assert reg_image.image.GetSpacing() == (0.65, 0.65)
    assert isinstance(reg_image.mask, itk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_RegImage_to_itk(im_gry_np, mask_np):
    reg_image = RegImage(im_gry_np, 0.65, mask=mask_np)
    reg_image.sitk_to_itk(cast_to_float32=True)
    assert isinstance(reg_image.image, itk.Image) is True
    assert reg_image.image.GetSpacing() == (0.65, 0.65)
    assert isinstance(reg_image.mask, itk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_mch")
def test_TransformRegImage_mc(disk_im_mch):
    reg_image = TransformRegImage("test", disk_im_mch, 0.65)
    assert np.array_equal(reg_image.im_dims, np.array([3, 2048, 2048])) is True
    assert reg_image.im_dtype == np.uint16
    assert reg_image.is_rgb == False


@pytest.mark.usefixtures("disk_im_rgb")
def test_TransformRegImage_rgb(disk_im_mch):
    reg_image = TransformRegImage("test", disk_im_mch, 0.65)
    assert np.array_equal(reg_image.im_dims, np.array([2048, 2048, 3])) is True
    assert reg_image.im_dtype == np.uint16
    assert reg_image.is_rgb == True


@pytest.mark.usefixtures("disk_im_rgb")
def test_TransformRegImage_rgb(disk_im_rgb):
    reg_image = TransformRegImage("test", disk_im_rgb, 0.65)
    assert np.array_equal(reg_image.im_dims, np.array([2048, 2048, 3])) is True
    assert reg_image.im_dtype == np.uint8
    assert reg_image.is_rgb == True


@pytest.mark.usefixtures("disk_im_rgb", "complex_transform")
def test_TransformRegImage_rgb_transform(
    tmpdir_factory, disk_im_rgb, complex_transform
):
    reg_image = TransformRegImage(
        "test", disk_im_rgb, 0.65, transform_data=complex_transform
    )
    out_dir = tmpdir_factory.mktemp("image")
    reg_image.transform_image(out_dir, "ome.tiff", tile_size=64)
    out_im_fp = f"{out_dir}/test.ome.tiff"
    out_im = imread(out_im_fp)
    out_im_zarr = imread(out_im_fp, aszarr=True)
    out_im_aszarr = zarr.open(out_im_zarr)

    assert len(reg_image.itk_transforms) == 6
    assert reg_image.composite_transform.GetNumberOfTransforms() == 6
    assert len(out_im.shape) == 3
    assert out_im.shape[2] == 3
    assert len(out_im_aszarr) == 4
    assert out_im.dtype == reg_image.im_dtype

@pytest.mark.usefixtures("disk_im_mch", "complex_transform")
def test_TransformRegImage_rgb_transform(
    tmpdir_factory, disk_im_mch, complex_transform
):
    reg_image = TransformRegImage(
        "test", disk_im_mch, 0.65, transform_data=complex_transform
    )
    out_dir = tmpdir_factory.mktemp("image")
    reg_image.transform_image(out_dir, "ome.tiff", tile_size=64)
    out_im_fp = f"{out_dir}/test.ome.tiff"
    out_im = imread(out_im_fp)
    out_im_zarr = imread(out_im_fp, aszarr=True)
    out_im_aszarr = zarr.open(out_im_zarr)

    assert len(out_im.shape) == 3
    assert out_im.shape[0] == 3
    assert len(out_im_aszarr) == 4
    assert out_im.dtype == reg_image.im_dtype

@pytest.mark.usefixtures("disk_im_gry", "complex_transform")
def test_TransformRegImage_rgb_transform(
    tmpdir_factory, disk_im_gry, complex_transform
):
    reg_image = TransformRegImage(
        "test", disk_im_gry, 0.65, transform_data=complex_transform
    )
    out_dir = tmpdir_factory.mktemp("image")
    reg_image.transform_image(out_dir, "ome.tiff", tile_size=64)
    out_im_fp = f"{out_dir}/test.ome.tiff"
    out_im = imread(out_im_fp)
    out_im_zarr = imread(out_im_fp, aszarr=True)
    out_im_aszarr = zarr.open(out_im_zarr)

    assert len(out_im.shape) == 2
    assert len(out_im_aszarr) == 4
    assert out_im.dtype == reg_image.im_dtype
