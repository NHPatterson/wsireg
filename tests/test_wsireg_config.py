import pytest
import os
from pathlib import Path
from wsireg.wsireg2d import WsiReg2D
from wsireg.utils.config_utils import parse_check_reg_config


HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
config1_fp = str(Path(FIXTURES_DIR) / "test-config1.yaml")
config2_fp = str(Path(FIXTURES_DIR) / "test-config2.yaml")


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
