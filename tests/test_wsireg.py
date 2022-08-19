import os
import random
import string
from pathlib import Path

import numpy as np
import pytest
from ome_types import from_xml
from tifffile import TiffFile, imread
import dask

from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.reg_images.loader import reg_image_loader
from wsireg.wsireg2d import WsiReg2D

HERE = os.path.dirname(__file__)
GEOJSON_FP = os.path.join(HERE, "fixtures/polygons.geojson")


def gen_project_name_str():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))


@pytest.fixture(scope="session")
def data_out_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("output")
    return out_dir


@pytest.fixture(scope="session")
def data_im_fp(tmpdir_factory):
    out_im = tmpdir_factory.mktemp("image").join("image_fp.tiff")
    return out_im


def test_WsiReg2D_instantiation(data_out_dir):
    pstr = gen_project_name_str()
    wsi_reg = WsiReg2D(pstr, str(data_out_dir))
    assert wsi_reg.project_name == pstr
    assert wsi_reg.output_dir == Path(str(data_out_dir))

dask.config.set(scheduler="single-threaded")
def test_wsireg2d_add_modality_w_fp(data_out_dir, data_im_fp):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
        preprocessing={
            "mask_bbox": [512, 512, 512, 512],
            "crop_to_mask_bbox": True,
        },
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
    assert registered_image_nocrop.shape[1:] == (2048, 2048)
    assert unregistered_image_nocrop.shape[1:] == (2048, 2048)
    assert registered_image_crop.shape[1:] == (512, 512)
    assert unregistered_image_crop.shape[1:] == (512, 512)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_with_flip_crop(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
        preprocessing={
            "mask_bbox": [512, 512, 512, 512],
            "flip": "h",
            "crop_to_mask_bbox": True,
        },
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

    assert registered_image_nocrop.shape[1:] == (2048, 2048)
    assert unregistered_image_nocrop.shape[1:] == (2048, 2048)
    assert registered_image_crop.shape[1:] == (512, 512)
    assert unregistered_image_crop.shape[1:] == (512, 512)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_with_crop_merge(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
        preprocessing={
            "mask_bbox": [512, 512, 512, 512],
            "crop_to_mask_bbox": True,
        },
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
    assert registered_image_nocrop.shape[1:] == (2048, 2048)
    assert registered_image_crop.shape[1:] == (512, 512)


def test_wsireg_run_reg_wmerge(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    assert merged_im.shape == (2, 2048, 2048)


def test_wsireg_run_reg_wmerge_and_indiv(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    assert merged_im.shape == (2, 2048, 2048)


def test_wsireg_run_reg_wattachment(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
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
    wsi_reg.add_attachment_images("mod1", "attached2", im1, image_res=0.65)

    wsi_reg.add_reg_path(
        "mod2", "mod1", reg_params=["rigid_test", "affine_test"]
    )

    wsi_reg.register_images()
    im_fps = wsi_reg.transform_images(transform_non_reg=False)

    wsi_reg.save_transformations()

    regim = reg_image_loader(im_fps[0], 0.65)
    attachim = reg_image_loader(im_fps[1], 0.65)
    attachim2 = reg_image_loader(im_fps[2], 0.65)

    assert np.array_equal(
        np.squeeze(regim.dask_image.compute()),
        np.squeeze(attachim.dask_image.compute()),
    )
    assert np.array_equal(
        np.squeeze(im1), np.squeeze(attachim2.dask_image.compute())
    )


def test_wsireg_run_reg_wattachment_ds2(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    im1 = np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)
    im2 = np.random.randint(0, 255, (2048, 2048), dtype=np.uint16)

    wsi_reg.add_modality(
        "mod1",
        im1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_modality(
        "mod2",
        im2,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )
    wsi_reg.add_attachment_images("mod2", "attached", im2, image_res=0.65)
    wsi_reg.add_attachment_images("mod1", "attached2", im1, image_res=0.65)

    wsi_reg.add_reg_path(
        "mod2", "mod1", reg_params=["rigid_test", "affine_test"]
    )

    wsi_reg.register_images()
    im_fps = wsi_reg.transform_images(transform_non_reg=False)

    wsi_reg.save_transformations()

    regim = reg_image_loader(im_fps[0], 0.65)
    attachim = reg_image_loader(im_fps[1], 0.65)
    attachim2 = reg_image_loader(im_fps[2], 0.65)

    assert np.array_equal(
        np.squeeze(regim.dask_image.compute()),
        np.squeeze(attachim.dask_image.compute()),
    )
    assert np.array_equal(
        np.squeeze(im1), np.squeeze(attachim2.dask_image.compute())
    )


@pytest.mark.usefixtures("im_gry_np")
def test_wsireg_run_reg_shapes(data_out_dir, im_gry_np):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = im_gry_np

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
    wsi_reg.add_attachment_shapes("mod1", "shapeset", GEOJSON_FP)
    wsi_reg.register_images()
    wsi_reg.transform_shapes()
    wsi_reg.save_transformations()
    im_fps = wsi_reg.transform_images(transform_non_reg=False)
    gj_files = sorted(Path(im_fps[0]).parent.glob("*.geojson"))

    assert Path(im_fps[0]).exists() is True
    assert len(gj_files) > 0


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_changeres(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        output_res=0.325,
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

    im_fps = wsi_reg.transform_images(transform_non_reg=False)
    regim = reg_image_loader(im_fps[0], 0.325)

    assert regim.shape[1:] == (4096, 4096)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
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

    im_fps = wsi_reg.transform_images(transform_non_reg=False)
    regim = reg_image_loader(im_fps[0], 0.65)

    assert regim.shape[1:] == (2048, 2048)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1_prepro(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2, "rot_cc": 90},
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

    im_fps = wsi_reg.transform_images(transform_non_reg=False)
    regim = reg_image_loader(im_fps[0], 0.65)

    assert regim.shape[1:] == (2048, 2048)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1m2(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.register_images()

    im_fps = wsi_reg.transform_images(transform_non_reg=False)
    regim = reg_image_loader(im_fps[0], 0.65)

    assert regim.shape[1:] == (2048, 2048)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1m2_changeores(
    data_out_dir, disk_im_gry
):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
        output_res=(1.3, 1.3),
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.register_images()

    im_fps = wsi_reg.transform_images(transform_non_reg=False)
    regim = reg_image_loader(im_fps[0], 0.65)

    assert regim.shape[1:] == (1024, 1024)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m2_prepro(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"rot_cc": 90, "downsampling": 2},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.register_images()

    im_fps = wsi_reg.transform_images(transform_non_reg=True)
    regim = reg_image_loader(im_fps[1], 0.65)

    assert regim.shape[1:] == (2048, 2048)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1m2_merge(data_out_dir, disk_im_gry):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2, "rot_cc": 90},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"rot_cc": 90, "downsampling": 2},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.add_merge_modalities("mod12-merge", ["mod1", "mod2"])
    wsi_reg.register_images()

    im_fps = wsi_reg.transform_images(
        transform_non_reg=True, remove_merged=True
    )
    regim = reg_image_loader(im_fps[0], 0.65)

    assert regim.shape == (2, 2048, 2048)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1m2_merge_no_prepro(
    data_out_dir, disk_im_gry
):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.add_merge_modalities("mod12-merge", ["mod1", "mod2"])
    wsi_reg.register_images()

    im_fps = wsi_reg.transform_images(
        transform_non_reg=False, remove_merged=True
    )
    regim = reg_image_loader(im_fps[0], 0.65)

    assert regim.shape == (2, 2048, 2048)


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_m1m2_merge_ds_attach(
    data_out_dir, disk_im_gry
):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_attachment_images(
        "mod2",
        "mod3",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])

    wsi_reg.add_merge_modalities("mod12-merge", ["mod1", "mod2", "mod3"])
    wsi_reg.register_images()

    im_fps = wsi_reg.transform_images(
        transform_non_reg=False, remove_merged=True
    )
    regim = reg_image_loader(im_fps[0], 0.65)
    ome_data = from_xml(TiffFile(im_fps[0]).ome_metadata)

    assert regim.shape == (3, 2048, 2048)
    assert ome_data.images[0].pixels.physical_size_x == 0.65
    assert ome_data.images[0].pixels.physical_size_y == 0.65
    assert ome_data.images[0].pixels.size_x == 2048
    assert ome_data.images[0].pixels.size_y == 2048
    assert ome_data.images[0].pixels.size_c == 3


@pytest.mark.usefixtures("disk_im_gry")
def test_wsireg_run_reg_downsampling_from_cache(data_out_dir, disk_im_gry):
    pstr = gen_project_name_str()
    wsi_reg = WsiReg2D(pstr, str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg.add_attachment_images(
        "mod2",
        "mod3",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.register_images()

    im_fps_nocache = wsi_reg.transform_images(
        transform_non_reg=True, remove_merged=True
    )

    regim_nocache = reg_image_loader(im_fps_nocache[0], 0.65)
    regim_nocache_attach = reg_image_loader(im_fps_nocache[1], 0.65)
    regim_nocache_br = reg_image_loader(im_fps_nocache[2], 0.65)

    wsi_reg2 = WsiReg2D(pstr, str(data_out_dir))
    img_fp1 = str(disk_im_gry)

    wsi_reg2.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg2.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"downsampling": 2},
    )

    wsi_reg2.add_attachment_images(
        "mod2",
        "mod3",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )
    wsi_reg2.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg2.register_images()

    im_fps_cache = wsi_reg.transform_images(
        transform_non_reg=True, remove_merged=True
    )

    regim_cache = reg_image_loader(im_fps_cache[0], 0.65)
    regim_cache_attach = reg_image_loader(im_fps_cache[1], 0.65)
    regim_cache_br = reg_image_loader(im_fps_cache[2], 0.65)

    assert regim_cache.shape == regim_nocache.shape
    assert regim_cache_br.shape == regim_nocache_br.shape
    assert regim_cache_attach.shape == regim_nocache_attach.shape


def test_wsireg_remove_modality(data_out_dir):
    pstr = gen_project_name_str()

    wsi_reg = WsiReg2D(pstr, str(data_out_dir))

    wsi_reg.add_modality("preAF-IMS", "", 0.65)
    wsi_reg.add_modality("preAF-MxIF", "", 0.65)
    wsi_reg.add_modality("mxif1", "", 0.65)
    wsi_reg.add_modality("mxif2", "", 0.65)
    wsi_reg.add_modality("mxif3", "", 0.65)
    wsi_reg.add_modality("pas", "", 0.65)

    wsi_reg.add_reg_path("preAF-MxIF", "preAF-IMS", reg_params=["rigid"])
    wsi_reg.add_reg_path(
        "mxif1", "preAF-IMS", thru_modality="preAF-MxIF", reg_params=["rigid"]
    )
    wsi_reg.add_reg_path(
        "mxif2", "preAF-IMS", thru_modality="mxif1", reg_params=["rigid"]
    )
    wsi_reg.add_reg_path(
        "mxif3", "preAF-IMS", thru_modality="mxif1", reg_params=["rigid"]
    )
    wsi_reg.add_reg_path("pas", "preAF-IMS", reg_params=["rigid"])

    wsi_reg.add_attachment_images("pas", "pas-attach", "", 0.65)
    wsi_reg.add_attachment_shapes("pas", "pas-shape", "")

    wsi_reg.add_merge_modalities("mxif-merge-1-2", ["mxif1", "mxif2"])
    assert len(wsi_reg.merge_modalities.keys()) == 1

    wsi_reg.remove_modality("pas")
    assert wsi_reg.n_registrations == 4
    assert wsi_reg.n_modalities == 6
    assert "PAS" not in wsi_reg.reg_paths.keys()

    wsi_reg.remove_modality("mxif1")
    assert wsi_reg.n_registrations == 1
    assert wsi_reg.n_modalities == 5
    assert "mxif1" not in wsi_reg.reg_paths.keys()
    assert len(wsi_reg.merge_modalities.keys()) == 0

    wsi_reg.remove_modality("pas-attach")
    assert wsi_reg.n_modalities == 4
    assert "pas-attach" not in wsi_reg.modality_names
    assert "pas-attach" not in wsi_reg.attachment_images.keys()

    wsi_reg.remove_modality("pas-shape")
    assert len(wsi_reg.shape_set_names) == 0
    assert wsi_reg.shape_sets.get("pas-shape") is None


@pytest.mark.usefixtures("im_mch_np")
def test_wsireg_run_reg_w_override(data_out_dir, im_mch_np):
    wsi_reg = WsiReg2D(gen_project_name_str(), str(data_out_dir))
    img_fp1 = im_mch_np

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
    wsi_reg.add_modality(
        "mod4",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.add_reg_path("mod4", "mod3", reg_params=["rigid_test"])
    wsi_reg.add_reg_path(
        "mod2",
        "mod3",
        reg_params=["rigid_test"],
        override_prepro={
            "source": {"ch_indices": [1]},
            "target": {"ch_indices": [1]},
        },
    )
    wsi_reg.register_images()

    or_mod2 = imread(wsi_reg.image_cache / "mod2-mod3-override_prepro.tiff")
    or_mod3 = imread(wsi_reg.image_cache / "mod3-mod2-override_prepro.tiff")
    pp_mod2 = imread(wsi_reg.image_cache / "mod2_prepro.tiff")
    pp_mod3 = imread(wsi_reg.image_cache / "mod3_prepro.tiff")

    assert not np.array_equal(or_mod2, pp_mod2)
    assert not np.array_equal(or_mod3, pp_mod3)


@pytest.mark.usefixtures("im_mch_np")
def test_wsireg_run_reg_reload_from_cache(data_out_dir, im_mch_np):
    img_fp1 = im_mch_np
    output_dir = str(data_out_dir)
    pname = gen_project_name_str()
    wsi_reg = WsiReg2D(pname, output_dir)

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

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])

    wsi_reg.register_images()

    pp_mod1_r1 = imread(wsi_reg.image_cache / "mod1_prepro.tiff")
    pp_mod2_r1 = imread(wsi_reg.image_cache / "mod2_prepro.tiff")

    # run registration again, loading data from cache
    wsi_reg = WsiReg2D(pname, output_dir)

    wsi_reg.add_modality(
        "mod1",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"ch_indices": [1]},
    )

    wsi_reg.add_modality(
        "mod2",
        img_fp1,
        0.65,
        channel_names=["test"],
        channel_colors=["red"],
        preprocessing={"ch_indices": [1]},
    )

    wsi_reg.add_reg_path("mod1", "mod2", reg_params=["rigid_test"])
    wsi_reg.register_images()

    pp_mod1_r2 = imread(wsi_reg.image_cache / "mod1_prepro.tiff")
    pp_mod2_r2 = imread(wsi_reg.image_cache / "mod2_prepro.tiff")

    assert not np.array_equal(pp_mod1_r1, pp_mod1_r2)
    assert not np.array_equal(pp_mod2_r1, pp_mod2_r2)
