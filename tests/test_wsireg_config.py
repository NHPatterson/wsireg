import pytest
from pathlib import Path
from wsireg.wsireg2d import WsiReg2D
from wsireg.utils.config_utils import parse_check_reg_config


@pytest.fixture(scope="session")
def data_out_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("output")
    return out_dir


def config_to_WsiReg2D(config_filepath, output_dir):
    reg_config = parse_check_reg_config(config_filepath)

    reg_graph = WsiReg2D(
        reg_config.get("project_name"),
        output_dir,
        reg_config.get("cache_images"),
    )
    return reg_graph


def test_wsireg_config1(data_out_dir):
    wsi_reg = config_to_WsiReg2D("./fixtures/test-config1.yaml", data_out_dir)
    wsi_reg.add_data_from_config("./fixtures/test-config1.yaml")
    wsi_reg.register_images()
    wsi_reg.save_transformations()
    assert wsi_reg.output_dir == Path(str(data_out_dir))


def test_wsireg_config2_chsel(data_out_dir):
    wsi_reg = config_to_WsiReg2D("./fixtures/test-config2.yaml", data_out_dir)
    wsi_reg.add_data_from_config("./fixtures/test-config2.yaml")
    wsi_reg.register_images()
    wsi_reg.save_transformations()
    assert wsi_reg.output_dir == Path(str(data_out_dir))


def test_wsireg_config1_fromcache(data_out_dir):
    wsi_reg1 = config_to_WsiReg2D("./fixtures/test-config1.yaml", data_out_dir)
    wsi_reg1.add_data_from_config("./fixtures/test-config1.yaml")
    wsi_reg1.register_images()

    wsi_reg2 = config_to_WsiReg2D("./fixtures/test-config1.yaml", data_out_dir)
    wsi_reg2.add_data_from_config("./fixtures/test-config1.yaml")
    wsi_reg2.register_images()

    assert wsi_reg1.output_dir == Path(str(data_out_dir))
