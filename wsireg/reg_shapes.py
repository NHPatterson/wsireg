import json
from wsireg.utils.shape_utils import (
    shape_reader,
    prepare_pt_transformation_data,
    transform_shapes,
    scale_shape_coordinates,
    insert_transformed_pts_gj,
)


class RegShapes:
    """
    Class that holds and manages shape data and it's transformations in the registration graph

    Parameters
    ----------
    shape_data: list
        list of np.ndarrays of shape data
        str - file path to GeoJSON file containing shape data
        np.ndarray - single shape
    itk_point_transforms: list
        list of ITK point transforms, appropriately inverted where non-linear.
        Usually generated on-the-fly when points are transformed from wsireg transformation data
    source_res:float
        isotropic image resolution of the source image in the registration stack,
        i.e., resolution of the image to which shape data is associated
    target_res:float
        isotropic image resolution of the target image in the registration stack
        usually generated on-the-fly when points are transformed from wsireg transformation data
    kwargs
        keyword arguments passed to wsireg.utils.tform_utils.shape_reader (shape_name, shape_type)
    """

    def __init__(
        self,
        shape_data=None,
        itk_point_transforms=None,
        source_res=1,
        target_res=1,
        **kwargs,
    ):
        self.shape_data = []
        self.shape_data_gj = []
        self.transformed_shape_data = []
        self.itk_point_transforms = itk_point_transforms
        self.source_res = source_res
        self.target_res = target_res

        self._n_shapes = None
        self._shape_types = None
        self._shape_names = None

        if shape_data is not None:
            self.add_shapes(shape_data, **kwargs)

    @property
    def n_shapes(self):
        """
        Number of shapes loaded
        """
        return self._n_shapes

    @property
    def shape_types(self):
        """
        List of GeoJSON shape types in shape data
        """
        return self._n_shape_types

    @property
    def shape_names(self):
        """
        list of GeoJSON shape names in shape data
        """
        return self._shape_names

    def add_shapes(self, shape_data, **kwargs):
        """
        Add shapes via shape_reader, will extend current shape list rather than overwrite it

        Parameters
        ----------
        shape_data
            list of np.ndarrays of shape data
            str - file path to GeoJSON file containing shape data
            np.ndarray - single shape
        """
        gj_shapes, np_shapes = shape_reader(shape_data, **kwargs)

        self.update_shapes(np_shapes)
        self.update_shapes_gj(gj_shapes)

    def update_shapes(self, imported_shapes):
        """
        Extend list of shape data with new shape data
        """
        self.shape_data.extend(imported_shapes)

    def update_shapes_gj(self, imported_shapes):
        """
        extend list of shape data in GeoJSON format and re-tabulate shape meta data
        """
        self.shape_data_gj.extend(imported_shapes)

        self._n_shapes = len(self.shape_data_gj)
        self._n_shape_types = [
            sh["geometry"]["type"] for sh in self.shape_data_gj
        ]
        self._shape_names = [
            sh["properties"]["classification"]["name"]
            for sh in self.shape_data_gj
        ]

    def scale_shapes(self, scale_factor):
        """
        scale coordinates of list of shape data by scale_factor

        Parameters
        ----------
        scale_factor: float
            isotropic scaling factor for the coordinates
        """
        self.shape_data = [
            scale_shape_coordinates(shape, scale_factor)
            for shape in self.shape_data
        ]

    def transform_shapes(self, transformations, px_idx=True, output_idx=True):
        """
        transform shapes using transformations data from wsireg

        Parameters
        ----------
        transformations
            list of dict containing elastix transformation data or str to wsireg .json file containing
            elastix transformation data
        px_idx: bool
            whether shape points are specified in physical coordinates (i.e., microns) or
            in pixel indices
        output_idx: bool
            whether transformed shape points should be output in physical coordinates (i.e., microns) or
            in pixel indices
        """
        if self.itk_point_transforms is None:
            (
                self.itk_point_transforms,
                self.target_res,
            ) = prepare_pt_transformation_data(transformations)

        self.transformed_shape_data = transform_shapes(
            self.shape_data,
            self.itk_point_transforms,
            px_idx=px_idx,
            source_res=self.source_res,
            output_idx=output_idx,
            target_res=self.target_res,
        )

    def save_shape_data(self, output_fp, transformed=True):
        """
        Save shape file to GeoJSON on disk.

        Parameters
        ----------
        output_fp: str
            path to the .json file where shape data will be saved
        transformed:bool
            save the transformed shape data or shape data as currently help in memory
        """
        if transformed is True:
            # updated GeoJSON with transformed points
            out_shapes = insert_transformed_pts_gj(
                self.shape_data_gj, self.transformed_shape_data
            )
        else:
            out_shapes = self.shape_data_gj

        json.dump(
            out_shapes,
            open(
                output_fp,
                "w",
            ),
            indent=1,
        )
