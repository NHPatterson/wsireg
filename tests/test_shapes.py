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


def test_RegShapes_shape_types_numpy():

    points = np.random.randint(1, 100, (2, 20))
    shape_data = []
    for point in points.transpose():
        shape_data.append(
            {
                "array": point,
                "shape_type": 'point',
                "shape_name": 'point',
            }
        )

    ri = RegShapes(shape_data)
    assert len(ri.shape_data_gj[0]["geometry"]["coordinates"]) == 2
    assert ri.shape_data_gj[0]["geometry"]["type"] == "Point"

    shape_data = [
        {
            "array": points,
            "shape_type": 'multipoint',
            "shape_name": 'multipoint',
        }
    ]

    ri = RegShapes(shape_data)
    assert len(ri.shape_data_gj[0]["geometry"]["coordinates"]) == 20
    assert ri.shape_data_gj[0]["geometry"]["type"] == "MultiPoint"

    shape_data = [
        {
            "array": points,
            "shape_type": 'linestring',
            "shape_name": 'linestring',
        }
    ]
    ri = RegShapes(shape_data)

    assert len(ri.shape_data_gj[0]["geometry"]["coordinates"]) == 20
    assert ri.shape_data_gj[0]["geometry"]["type"] == "LineString"


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
