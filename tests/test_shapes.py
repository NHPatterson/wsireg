import os
from copy import deepcopy

import numpy as np
import pytest

from wsireg.reg_shapes import RegShapes

HERE = os.path.dirname(__file__)
GEOJSON_FP = os.path.join(HERE, "fixtures/polygons.geojson")


@pytest.mark.usefixtures("complex_transform")
def test_RegShapes_transform(complex_transform):
    rs = RegShapes(GEOJSON_FP)
    shape0 = deepcopy(rs.shape_data[0])
    rs.transform_shapes(complex_transform)
    shape0_tform = rs.transformed_shape_data[0]
    assert np.array_equal(shape0["array"], shape0_tform["array"]) is False


def test_RegShapes_drawmask():
    triangles = [
        np.array([[11, 13], [111, 113], [22, 246]]),
        np.array([[11, 13], [111, 113], [22, 246]]) * 2,
    ]
    rs = RegShapes(triangles)

    bin_mask = rs.draw_mask((512, 512), labels=False)
    assert np.sum(bin_mask) > 0

    lab_mask = rs.draw_mask((512, 512), labels=True)

    # 0 = bg
    assert len(np.unique(lab_mask)) == 3
