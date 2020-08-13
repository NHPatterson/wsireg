import geojson
from wsireg.shape_utils import (
    shape_reader,
    apply_transformation_dict_shapes,
)


class RegShapes:
    def __init__(self, shape_data=None, **kwargs):
        self.shape_data = []
        self.transformed_shape_data = []
        self._n_shapes = None
        self._shape_types = None
        self._shape_names = None

        if shape_data is not None:
            self.add_shapes(shape_data, **kwargs)

    @property
    def n_shapes(self):
        return self._n_shapes

    @property
    def shape_types(self):
        return self._n_shape_types

    @property
    def shape_names(self):
        return self._shape_names

    def add_shapes(self, shape_data, **kwargs):

        imported_shapes = shape_reader(shape_data, **kwargs)

        self.update_shapes(imported_shapes)

    def update_shapes(self, imported_shapes):
        self.shape_data.extend(imported_shapes)

        self._n_shapes = len(self.shape_data)
        self._n_shape_types = [
            sh["geometry"]["type"] for sh in self.shape_data
        ]
        self._shape_names = [
            sh["properties"]["classification"]["name"]
            for sh in self.shape_data
        ]

    def transform_shapes(self, image_res, tform_dict):

        self.transformed_shapes = apply_transformation_dict_shapes(
            self.shape_data, image_res, tform_dict
        )

    def save_shape_data(self, output_fp, transformed=True):
        if transformed is True:
            out_shapes = self.transformed_shape_data
        else:
            out_shapes = self.shape_data

        geojson.dump(
            out_shapes, open(output_fp, "w",), indent=4,
        )
