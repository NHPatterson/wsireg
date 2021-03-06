import pytest
from wsireg.reg_images.loader import reg_image_loader
import SimpleITK as sitk
import itk


@pytest.mark.usefixtures("disk_im_mch")
def test_reg_image_loader_image_fp_mc_std_prepro(disk_im_mch):
    reg_image = reg_image_loader(str(disk_im_mch), 0.65)
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_rgb")
def test_reg_image_loader_image_fp_rgb_std_prepro(disk_im_rgb):
    reg_image = reg_image_loader(str(disk_im_rgb), 0.65)
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("disk_im_gry")
def test_reg_image_loader_image_fp_gry_std_prepro(disk_im_gry):
    reg_image = reg_image_loader(str(disk_im_gry), 0.65)
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65)
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65)
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65)
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_rot(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65, preprocessing={"rot_cc": 90})
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro_rot(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65, preprocessing={"rot_cc": 90})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_rot(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"rot_cc": 90})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_fliph(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65, preprocessing={"flip": "h"})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro_fliph(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65, preprocessing={"flip": "h"})
    reg_image.read_reg_image()
    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_fliph(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"flip": "h"})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_flipv(im_mch_np):
    reg_image = reg_image_loader(im_mch_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_rgb_np")
def test_reg_image_loader_image_np_rgb_std_prepro_flipv(im_rgb_np):
    reg_image = reg_image_loader(im_rgb_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_flipv(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_mch_np")
def test_reg_image_loader_image_np_mc_std_prepro_rot_flipv(im_mch_np):
    reg_image = reg_image_loader(
        im_mch_np, 0.65, preprocessing={"rot_cc": 90, "flip": "v"}
    )
    reg_image.read_reg_image()

    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.image.GetDepth() == 0
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 2
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np")
def test_reg_image_loader_image_np_gry_std_prepro_flipv(im_gry_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, preprocessing={"flip": "v"})
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (2048, 2048)
    assert reg_image.image.GetNumberOfComponentsPerPixel() == 1
    assert reg_image.pre_reg_transforms is not None
    assert len(reg_image.pre_reg_transforms) == 1
    assert reg_image.image.GetSpacing() == (0.65, 0.65)


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
        im_gry_np, 1, preprocessing={"downsample": 2}, mask=mask_np
    )
    reg_image.read_reg_image()

    assert reg_image.image.GetSize() == (1024, 1024)
    assert reg_image.mask.GetSize() == (1024, 1024)
    assert reg_image.mask.GetSpacing() == (2, 2)
    assert reg_image.image.GetSpacing() == (2, 2)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_to_itk(im_gry_np, mask_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, mask=mask_np)
    reg_image.read_reg_image()
    reg_image.sitk_to_itk(cast_to_float32=True)
    assert isinstance(reg_image.image, itk.Image) is True
    assert reg_image.image.GetSpacing() == (0.65, 0.65)
    assert isinstance(reg_image.mask, itk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)


@pytest.mark.usefixtures("im_gry_np", "mask_np")
def test_reg_image_loader_to_itk(im_gry_np, mask_np):
    reg_image = reg_image_loader(im_gry_np, 0.65, mask=mask_np)
    reg_image.read_reg_image()
    reg_image.sitk_to_itk(cast_to_float32=True)
    assert isinstance(reg_image.image, itk.Image) is True
    assert reg_image.image.GetSpacing() == (0.65, 0.65)
    assert isinstance(reg_image.mask, itk.Image) is True
    assert reg_image.mask.GetSpacing() == (0.65, 0.65)
