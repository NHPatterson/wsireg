import itk
import pytest
import SimpleITK as sitk

from wsireg.reg_images.loader import reg_image_loader


@pytest.mark.usefixtures("disk_im_mch")
def test_reg_image_loader_image_fp_mc_std_prepro(disk_im_mch):
    reg_image = reg_image_loader(str(disk_im_mch), 0.65)
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_rgb")
def test_reg_image_loader_image_fp_rgb_std_prepro(disk_im_rgb):
    reg_image = reg_image_loader(str(disk_im_rgb), 0.65)
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_gry")
def test_reg_image_loader_image_fp_gry_std_prepro(disk_im_gry):
    reg_image = reg_image_loader(str(disk_im_gry), 0.65)
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65)
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetDepth() == 0


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65)
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65)
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_rot(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65, preprocessing={"rot_cc": 90})
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro_rot(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65, preprocessing={"rot_cc": 90})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_rot(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"rot_cc": 90})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_fliph(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65, preprocessing={"flip": "h"})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro_fliph(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65, preprocessing={"flip": "h"})
    reg_image.read_reg_image()
    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_fliph(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"flip": "h"})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_flipv(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro_flipv(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_flipv(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_rot_flipv(im_mch_np):
    reg_image = reg_image_loader(
        im_mch_np, 0.65, preprocessing={"rot_cc": 90, "flip": "v"}
    )
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.reg_image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 2
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_flipv(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (2048, 2048)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_mask(im_gry_np, mask_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, mask=mask_np)
    reg_image.read_reg_image()
    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_mask_rot(im_gry_np, mask_np):
    reg_image = reg_image_loader(
        im_gry_np, 0.65, preprocessing={"rot_cc": 90}, mask=mask_np
    )
    reg_image.read_reg_image()

    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_mask_rot_flip(im_gry_np, mask_np):
    reg_image = reg_image_loader(
        im_gry_np,
        0.65,
        preprocessing={"rot_cc": 90, "flip": "h"},
        mask=mask_np,
    )
    reg_image.read_reg_image()

    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_mask_flip(im_gry_np, mask_np):
    reg_image = reg_image_loader(
        im_gry_np, 0.65, preprocessing={"flip": "v"}, mask=mask_np
    )
    reg_image.read_reg_image()

    assert reg_image.mask is not None
    assert isinstance(reg_image.mask, sitk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_downsampling(im_gry_np, mask_np):
    reg_image = reg_image_loader(
        im_gry_np, 1, preprocessing={"downsampling": 2}, mask=mask_np
    )
    reg_image.read_reg_image()

    assert reg_image.reg_image.GetSize() == (1024, 1024)
    assert reg_image.mask.GetSize() == (1024, 1024)
    assert reg_image.mask.GetSpacing() == (2, 2)
    assert reg_image.reg_image.GetSpacing() == (2, 2)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_to_itk(im_gry_np, mask_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, mask=mask_np)
    reg_image.read_reg_image()
    reg_image.reg_image_sitk_to_itk(cast_to_float32=True)
    assert isinstance(reg_image.reg_image, itk.Image) is True
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert isinstance(reg_image.mask, itk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_to_itk(im_gry_np, mask_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, mask=mask_np)
    reg_image.read_reg_image()
    reg_image.reg_image_sitk_to_itk(cast_to_float32=True)
    assert isinstance(reg_image.reg_image, itk.Image) is True
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert isinstance(reg_image.mask, itk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("dask_im_rgb_np")
def test_reg_image_loader_dask_rgb(dask_im_rgb_np):
    reg_image = reg_image_loader(dask_im_rgb_np, 0.65)
    reg_image.read_reg_image()
    assert len(reg_image.im_dims) == 3
    assert reg_image.im_dims[-1] == 3
    assert reg_image.is_rgb
    assert reg_image.n_ch == 3
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("dask_im_gry_np")
def test_reg_image_loader_dask_gry(dask_im_gry_np):
    reg_image = reg_image_loader(dask_im_gry_np, 0.65)
    reg_image.read_reg_image()
    assert len(reg_image.im_dims) == 3
    assert reg_image.im_dims[0] == 1
    assert reg_image.is_rgb is False
    assert reg_image.n_ch == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("dask_im_mch_np")
def test_reg_image_loader_dask_mch(dask_im_mch_np):
    reg_image = reg_image_loader(dask_im_mch_np, 0.65)
    reg_image.read_reg_image()
    assert len(reg_image.im_dims) == 3
    assert reg_image.im_dims[0] == 3
    assert reg_image.is_rgb is False
    assert reg_image.n_ch == 3
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("zarr_im_rgb_np")
def test_reg_image_loader_zarr_rgb(zarr_im_rgb_np):
    reg_image = reg_image_loader(zarr_im_rgb_np, 0.65)
    reg_image.read_reg_image()
    assert len(reg_image.im_dims) == 3
    assert reg_image.im_dims[-1] == 3
    assert reg_image.is_rgb
    assert reg_image.n_ch == 3
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("zarr_im_gry_np")
def test_reg_image_loader_zarr_gry(zarr_im_gry_np):
    reg_image = reg_image_loader(zarr_im_gry_np, 0.65)
    reg_image.read_reg_image()
    assert len(reg_image.im_dims) == 3
    assert reg_image.im_dims[0] == 1
    assert reg_image.is_rgb is False
    assert reg_image.n_ch == 1
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("zarr_im_mch_np")
def test_reg_image_loader_zarr_mch(zarr_im_mch_np):
    reg_image = reg_image_loader(zarr_im_mch_np, 0.65)
    reg_image.read_reg_image()
    assert len(reg_image.im_dims) == 3
    assert reg_image.im_dims[0] == 3
    assert reg_image.is_rgb is False
    assert reg_image.n_ch == 3
    assert reg_image.reg_image.GetSpacing() == (0.65, 0.65)
    assert reg_image.reg_image.GetNumberOfComponentsPerPixel() == 1
