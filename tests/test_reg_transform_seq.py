import os
from pathlib import Path

import pytest

from wsireg.reg_transform_seq import RegTransformSeq

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")


@pytest.mark.usefixtures("complex_transform_larger")
def test_RegTransformSeq_from_dict(complex_transform_larger):
    rts = RegTransformSeq(complex_transform_larger)

    assert len(rts.reg_transforms) == 6
    assert len(rts.reg_transforms_itk_order) == 6
    assert rts.transform_seq_idx == [0, 1, 2, 2, 3, 3]


@pytest.mark.usefixtures("complex_transform_larger")
def test_RegTransformSeq_resize_output_up(complex_transform_larger):
    rts = RegTransformSeq(complex_transform_larger)
    os_pre = rts.output_size
    rts.set_output_spacing((1, 1))

    assert rts.output_size == (2048, 2048)
    assert os_pre != rts.output_size


@pytest.mark.usefixtures("complex_transform_larger")
def test_RegTransformSeq_resize_output_down(complex_transform_larger):
    rts = RegTransformSeq(complex_transform_larger)
    os_pre = rts.output_size
    rts.set_output_spacing((4, 4))

    assert rts.output_size == (512, 512)
    assert os_pre != rts.output_size


def test_RegTransformSeq_from_json():
    test_tform = str(Path(FIXTURES_DIR) / "test-tform.json")
    rts = RegTransformSeq(test_tform)

    assert len(rts.reg_transforms) == 10


def test_RegTransformSeq_from_RegTransform():
    test_tform = str(Path(FIXTURES_DIR) / "test-tform.json")
    rts = RegTransformSeq(test_tform)
    rts_1 = RegTransformSeq(
        reg_transforms=[
            rts.reg_transforms[0],
            rts.reg_transforms[1],
            rts.reg_transforms[2],
        ],
        transform_seq_idx=[0, 1, 1],
    )
    assert len(rts_1.reg_transforms) == 3
    assert rts_1.transform_seq_idx == [0, 1, 1]


def test_RegTransformSeq_append():
    test_tform = str(Path(FIXTURES_DIR) / "test-tform.json")
    rts = RegTransformSeq(test_tform)
    rts_1 = RegTransformSeq(
        reg_transforms=[
            rts.reg_transforms[0],
            rts.reg_transforms[1],
            rts.reg_transforms[2],
        ],
        transform_seq_idx=[0, 1, 1],
    )
    rts_2 = RegTransformSeq(
        reg_transforms=[
            rts.reg_transforms[0],
            rts.reg_transforms[1],
            rts.reg_transforms[2],
        ],
        transform_seq_idx=[0, 1, 1],
    )
    rts_1.append(rts_2)

    assert len(rts_1.reg_transforms) == 6
    assert rts_1.transform_seq_idx == [0, 1, 1, 2, 3, 3]
