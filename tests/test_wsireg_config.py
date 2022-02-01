import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pytest
import SimpleITK as sitk

from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_shapes import RegShapes
from wsireg.utils.config_utils import parse_check_reg_config
from wsireg.wsireg2d import WsiReg2D

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
PRIVATE_DIR = os.path.join(HERE, "private_data")

config1_fp = str(Path(FIXTURES_DIR) / "test-config1.yaml")
config2_fp = str(Path(FIXTURES_DIR) / "test-config2.yaml")
config3_fp = str(Path(FIXTURES_DIR) / "test-config3.yaml")
config4_fp = str(Path(FIXTURES_DIR) / "test-config4.yaml")
config5_fp = str(Path(FIXTURES_DIR) / "test-config5.yaml")

SKIP_PRIVATE = False
REASON = "private data"

if not os.path.exists(PRIVATE_DIR):
    SKIP_PRIVATE = True


def geojson_to_binary_image(geojson_fp: Union[str, Path]) -> sitk.Image:
    rs = RegShapes(geojson_fp)
    shape_cv2 = [s["array"].astype(np.int32) for s in rs.shape_data]

    x_max = np.max([np.max(s["array"][:, 0]) for s in rs.shape_data]).astype(
        int
    )
    y_max = np.max([np.max(s["array"][:, 1]) for s in rs.shape_data]).astype(
        int
    )

    binary_image = np.zeros((y_max + 200, x_max + 200), dtype=np.uint8)
    return sitk.GetImageFromArray(cv2.fillPoly(binary_image, shape_cv2, 255))


def compute_dice(b1: sitk.Image, b2: sitk.Image):

    b2 = sitk.Resample(
        b2,
        b1.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        b1.GetOrigin(),
        b1.GetSpacing(),
        b1.GetDirection(),
        0,
        b2.GetPixelID(),
    )
    labstats = sitk.LabelOverlapMeasuresImageFilter()

    labstats.Execute(b1, b2)

    return labstats.GetDiceCoefficient()


def config_to_WsiReg2D(config_filepath, output_dir):
    reg_config = parse_check_reg_config(config_filepath)

    reg_graph = WsiReg2D(
        reg_config.get("project_name"),
        output_dir,
        reg_config.get("cache_images"),
    )
    return reg_graph


@pytest.fixture(scope="session")
def data_out_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("output")
    return out_dir


@pytest.mark.parametrize("config_fp", [(config1_fp), (config2_fp)])
def test_wsireg_configs(config_fp, data_out_dir):
    wsi_reg = config_to_WsiReg2D(config_fp, data_out_dir)
    wsi_reg.add_data_from_config(config_fp)
    wsi_reg.register_images()
    wsi_reg.save_transformations()
    assert wsi_reg.output_dir == Path(str(data_out_dir))


@pytest.mark.parametrize("config_fp", [(config1_fp), (config2_fp)])
def test_wsireg_configs_fromcache(config_fp, data_out_dir):
    wsi_reg1 = config_to_WsiReg2D(config_fp, data_out_dir)
    wsi_reg1.add_data_from_config(config_fp)
    wsi_reg1.register_images()

    wsi_reg2 = config_to_WsiReg2D(config_fp, data_out_dir)
    wsi_reg2.add_data_from_config(config_fp)
    wsi_reg2.register_images()

    assert wsi_reg1.output_dir == Path(str(data_out_dir))


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
@pytest.mark.parametrize(
    "config_fp",
    [
        (config3_fp),
    ],
)
def test_wsireg_config_full_exp_DICE(config_fp, data_out_dir):
    wsi_reg1 = config_to_WsiReg2D(config_fp, data_out_dir)
    wsi_reg1.add_data_from_config(config_fp)
    wsi_reg1.register_images()
    shape_fps = wsi_reg1.transform_shapes()
    gt = geojson_to_binary_image(
        "private_data/unreg_rois/VAN0006-LK-2-85-AF_preIMS_unregistered.geojson"
    )

    dice_vals = []
    for shape in shape_fps:
        test_mask = geojson_to_binary_image(shape)
        dice_vals.append(compute_dice(gt, test_mask))

    assert all(np.asarray(dice_vals) > 0.85)


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
@pytest.mark.parametrize(
    "config_fp",
    [
        (config4_fp),
    ],
)
def test_wsireg_config_full_exp_DICE_ds(config_fp, data_out_dir):
    wsi_reg1 = config_to_WsiReg2D(config_fp, data_out_dir)
    wsi_reg1.add_data_from_config(config_fp)
    wsi_reg1.register_images()
    shape_fps = wsi_reg1.transform_shapes()
    gt = geojson_to_binary_image(
        "private_data/unreg_rois/VAN0006-LK-2-85-AF_preIMS_unregistered.geojson"
    )

    dice_vals = []
    for shape in shape_fps:
        test_mask = geojson_to_binary_image(shape)
        dice_vals.append(compute_dice(gt, test_mask))

    assert all(np.asarray(dice_vals) > 0.8)


@pytest.mark.skipif(SKIP_PRIVATE, reason=REASON)
@pytest.mark.parametrize(
    "config_fp",
    [
        (config5_fp),
    ],
)
def test_wsireg_config_full_merge_rgb_mc(config_fp, data_out_dir):
    wsi_reg1 = config_to_WsiReg2D(config_fp, data_out_dir)
    wsi_reg1.add_data_from_config(config_fp)
    wsi_reg1.register_images()
    im_fps = wsi_reg1.transform_images()
    ri = reg_image_loader(im_fps[0], 1)

    assert ri.im_dtype == np.uint16
    assert ri.im_dims == (9, 3993, 3397)
