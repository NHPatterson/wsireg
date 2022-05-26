import os
from pathlib import Path


HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
PRIVATE_DIR = os.path.join(HERE, "private_data")

config1_fp = str(Path(FIXTURES_DIR) / "test-config1-cmd-line.yaml")


def test_wsireg2d_entrypoint():
    exit_status = os.system('wsireg2d --help')
    assert exit_status == 0


def test_wsireg2d_run():
    exit_status = os.system(f'wsireg2d "{str(config1_fp)}" --testing')
    assert exit_status == 0
