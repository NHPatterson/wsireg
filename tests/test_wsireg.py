import pytest
import numpy as np
from pathlib import Path
from wsireg.wsireg2d import WsiReg2D
from wsireg.reg_images.loader import reg_image_loader
from wsireg.parameter_maps.preprocessing import ImagePreproParams


@pytest.fixture(scope="session")
def data_out_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("output")
    return out_dir


@pytest.fixture(scope="session")
def data_im_fp(tmpdir_factory):
    out_im = tmpdir_factory.mktemp("image").join("image_fp.tiff")
    return out_im


def test_WsiReg2D_instantiation(data_out_dir):
    wsi_reg = WsiReg2D("test_proj", str(data_out_dir))
    assert wsi_reg.project_name == "test_proj"
    assert wsi_reg.output_dir == Path(str(data_out_dir))


def test_wsireg2d_add_modality_w_fp(data_out_dir, data_im_fp):
    wsi_reg = WsiReg2D("test_proj1", str(data_out_dir))
    img_fp1 = str(data_im_fp)
    wsi_reg.add_modality(
        "test_mod",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    assert wsi_reg.modalities.get("test_mod").get("image_filepath") == img_fp1
    assert wsi_reg.modalities.get("test_mod").get("image_res") == 0.65
    assert wsi_reg.modalities.get("test_mod").get("channel_names") == ["test"]
    assert wsi_reg.modalities.get("test_mod").get("channel_colors") == ["red"]
    assert (
        wsi_reg.modalities.get("test_mod").get("preprocessing")
        == ImagePreproParams()
    )
    assert wsi_reg.modalities.get("test_mod").get("mask") == None
    assert wsi_reg.n_modalities == 1


@pytest.mark.usefixtures("im_mch_np")
def test_wsireg2d_add_modality_w_np(data_out_dir, im_mch_np):
    wsi_reg = WsiReg2D("test_proj2", str(data_out_dir))
    wsi_reg.add_modality(
        "test_mod",
        im_mch_np,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    assert np.array_equal(
        wsi_reg.modalities.get("test_mod").get("image_filepath"), im_mch_np
    )
    assert wsi_reg.modalities.get("test_mod").get("image_res") == 0.65
    assert wsi_reg.modalities.get("test_mod").get("channel_names") == ["test"]
    assert wsi_reg.modalities.get("test_mod").get("channel_colors") == ["red"]
    assert (
        wsi_reg.modalities.get("test_mod").get("preprocessing")
        == ImagePreproParams()
    )
    assert wsi_reg.modalities.get("test_mod").get("mask") == None
    assert wsi_reg.n_modalities == 1


def test_wsireg2d_add_modality_check_names(data_out_dir, data_im_fp):
    wsi_reg = WsiReg2D("test_proj3", str(data_out_dir))
    img_fp1 = str(data_im_fp)
    modality_name1 = "test_mod1"
    modality_name2 = "test_mod2"
    wsi_reg.add_modality(
        modality_name1,
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg.add_modality(
        modality_name2,
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    assert modality_name1 in wsi_reg.modality_names
    assert modality_name2 in wsi_reg.modality_names
    assert wsi_reg.modality_names == ["test_mod1", "test_mod2"]
    assert wsi_reg.n_modalities == 2


def test_wsireg2d_add_reg_path_single(data_out_dir, data_im_fp):
    wsi_reg = WsiReg2D("test_proj4", str(data_out_dir))
    img_fp1 = str(data_im_fp)
    modality_name1 = "test_mod1"
    modality_name2 = "test_mod2"
    wsi_reg.add_modality(
        modality_name1,
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg.add_modality(
        modality_name2,
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg.add_reg_path(modality_name1, modality_name2, reg_params=["rigid"])
    assert wsi_reg.n_registrations == 1
    assert wsi_reg.reg_paths.get(modality_name1) == [modality_name2]
    assert (
        wsi_reg.reg_graph_edges[0].get("modalities").get("target")
        == modality_name2
    )


def test_wsireg2d_add_modality_duplicated_error(data_out_dir, data_im_fp):
    wsi_reg = WsiReg2D("test_proj5", str(data_out_dir))
    img_fp1 = str(data_im_fp)
    wsi_reg.add_modality(
        "test_mod",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    with pytest.raises(ValueError):
        wsi_reg.add_modality(
            "test_mod",
            img_fp1,
            0.65,
            channel_names=["test"],
            channel_colors=["red"],
        )


def test_wsireg2d_add_merge_modality_notfound(data_out_dir, data_im_fp):
    wsi_reg = WsiReg2D("test_proj_merge_nf", str(data_out_dir))
    img_fp1 = str(data_im_fp)
    wsi_reg.add_modality(
        "test_mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    img_fp1 = str(data_im_fp)
    wsi_reg.add_modality(
        "test_mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    with pytest.raises(ValueError):
        wsi_reg.add_merge_modalities("mergetest", ["test_mod1", "test_mod3"])


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj6", str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_reg_path(
        "mod1", "mod2", reg_params=["rigid_test", "affine_test"]
    )
    wsi_reg.register_images()
    wsi_reg.save_transformations()
    im_fps = wsi_reg.transform_images(transform_non_reg=True)

    assert Path(im_fps[0]).exists() is True
    assert Path(im_fps[1]).exists() is True


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_with_crop(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj7", str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"mask_bbox": [512, 512, 512, 512]},
    )

    wsi_reg.add_reg_path(
        "mod1", "mod2", reg_params=["rigid_test", "affine_test"]
    )
    wsi_reg.register_images()
    # not cropped
    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, to_original_size=True
    )

    registered_image_nocrop = reg_image_loader(im_fps[0], 1)
    unregistered_image_nocrop = reg_image_loader(im_fps[1], 1)

    # crop image
    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, to_original_size=False
    )
    wsi_reg.save_transformations()

    #
    registered_image_crop = reg_image_loader(im_fps[0], 1)
    unregistered_image_crop = reg_image_loader(im_fps[1], 1)
    #
    assert registered_image_nocrop.im_dims[1:] == (2048, 2048)
    assert unregistered_image_nocrop.im_dims[1:] == (2048, 2048)
    assert registered_image_crop.im_dims[1:] == (512, 512)
    assert unregistered_image_crop.im_dims[1:] == (512, 512)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_with_flip_crop(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj8", str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"mask_bbox": [512, 512, 512, 512], "flip": "h"},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.register_images()
    # not cropped
    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, to_original_size=True
    )
    registered_image_nocrop = reg_image_loader(im_fps[0], 1)
    unregistered_image_nocrop = reg_image_loader(im_fps[1], 1)

    # crop image
    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, to_original_size=False
    )
    wsi_reg.save_transformations()

    registered_image_crop = reg_image_loader(im_fps[0], 1)
    unregistered_image_crop = reg_image_loader(im_fps[1], 1)

    assert registered_image_nocrop.im_dims[1:] == (2048, 2048)
    assert unregistered_image_nocrop.im_dims[1:] == (2048, 2048)
    assert registered_image_crop.im_dims[1:] == (512, 512)
    assert unregistered_image_crop.im_dims[1:] == (512, 512)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_with_crop_merge(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj9", str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod3",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"mask_bbox": [512, 512, 512, 512]},
    )

    wsi_reg.add_reg_path("mod1", "mod3", reg_params=["rigid_test"])
    wsi_reg.add_reg_path("mod2", "mod3", reg_params=["rigid_test"])
    wsi_reg.register_images()
    wsi_reg.add_merge_modalities("merge", ["mod1", "mod2", "mod3"])
    # not cropped
    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, to_original_size=True
    )
    registered_image_nocrop = reg_image_loader(im_fps[0], 1)

    # crop image
    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, to_original_size=False
    )
    wsi_reg.save_transformations()
    registered_image_crop = reg_image_loader(im_fps[0], 1)
    assert registered_image_nocrop.im_dims[1:] == (2048, 2048)
    assert registered_image_crop.im_dims[1:] == (512, 512)


def test_wsireg_run_reg_wmerge(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj-merge-work", str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_reg_path(
        "mod1", "mod2", reg_params=["rigid_test", "affine_test"]
    )
    wsi_reg.add_merge_modalities("test_merge", ["mod1", "mod2"])
    wsi_reg.register_images()
    wsi_reg.save_transformations()
    im_fps = wsi_reg.transform_images(transform_non_reg=True)
    merged_im = reg_image_loader(im_fps[0], 0.65)
    assert Path(im_fps[0]).exists() is True
    assert merged_im.im_dims == (2, 2048, 2048)


def test_wsireg_run_reg_wmerge_and_indiv(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj-merge-and-unmerge", str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod3",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_reg_path(
        "mod2", "mod1", reg_params=["rigid_test", "affine_test"]
    )
    wsi_reg.add_reg_path(
        "mod3", "mod1", reg_params=["rigid_test", "affine_test"]
    )
    wsi_reg.add_merge_modalities("test_merge", ["mod1", "mod2"])
    wsi_reg.register_images()
    im_fps = wsi_reg.transform_images(
        remove_merged=False, transform_non_reg=True
    )

    merged_im = reg_image_loader(im_fps[-2], 0.65)
    wsi_reg.save_transformations()

    assert len(im_fps) == 4
    assert Path(im_fps[0]).exists() is True
    assert Path(im_fps[1]).exists() is True
    assert Path(im_fps[2]).exists() is True
    assert Path(im_fps[3]).exists() is True
    assert merged_im.im_dims == (2, 2048, 2048)


def test_wsireg_run_reg_wattachment(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D("test_proj-attach", str(data_out_dir))
    im1 = np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)
    im2 = np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)

    wsi_reg.add_modality(
        "mod1",
        im1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_modality(
        "mod2",
        im2,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg.add_attachment_images("mod2", "attached", im2, image_res=0.65)

    wsi_reg.add_reg_path(
        "mod2", "mod1", reg_params=["rigid_test", "affine_test"]
    )

    wsi_reg.register_images()
    im_fps = wsi_reg.transform_images(transform_non_reg=False)

    wsi_reg.save_transformations()

    regim = reg_image_loader(im_fps[0], 0.65)
    attachim = reg_image_loader(im_fps[1], 0.65)

    assert np.array_equal(regim.image.compute(), attachim.image.compute())
