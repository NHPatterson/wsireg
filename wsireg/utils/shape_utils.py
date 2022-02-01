import json
import zipfile
from copy import deepcopy
from pathlib import Path

import cv2
import geojson
import numpy as np
import SimpleITK as sitk

from wsireg.reg_transform import RegTransform
from wsireg.utils.tform_utils import wsireg_transforms_to_itk_composite

GJ_SHAPE_TYPE = {
    "polygon": geojson.Polygon,
    "multipolygon": geojson.MultiPolygon,
    "point": geojson.Point,
    "multipoint": geojson.MultiPoint,
    "multilinestring": geojson.MultiLineString,
    "linestring": geojson.LineString,
}


def gj_to_np(gj: dict):
    """
    Convert geojson representation to np.ndarray representation of shape

    Parameters
    ----------
    gj : dict
        GeoJSON data stored as python dict

    Returns
    -------
    dict
        containing keys
            "array": np.ndarray - x,y point data in array
            "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
            "shape_name": str - name inherited from QuPath GeoJSON
    """
    if gj.get("geometry").get("type") == "MultiPolygon":
        pts = []
        for geo in gj.get("geometry").get("coordinates"):
            pts.append(np.squeeze(np.array(geo)))
        pts = np.vstack(pts)

    elif gj.get("geometry").get("type") == "Polygon":
        pts = np.squeeze(np.asarray(gj.get("geometry").get("coordinates")))
    elif gj.get("geometry").get("type") == "Point":
        pts = np.expand_dims(
            np.asarray(gj.get("geometry").get("coordinates")), 0
        )
    elif gj.get("geometry").get("type") == "MultiPoint":
        pts = np.asarray(gj.get("geometry").get("coordinates"))
    elif gj.get("geometry").get("type") == "LineString":
        pts = np.asarray(gj.get("geometry").get("coordinates"))

    if gj.get("properties").get("classification") is None:
        shape_name = "unnamed"
    else:
        shape_name = gj.get("properties").get("classification").get("name")

    return {
        "array": pts.astype(np.double),
        "shape_type": gj.get("geometry").get("type"),
        "shape_name": shape_name,
    }


def add_unamed(gj):
    if gj.get("properties").get("classification") is None:
        gj.get("properties").update({"classification": {"name": "unnamed"}})
    return gj


def read_geojson(json_file: str):
    """Read GeoJSON files (and some QuPath metadata).

    Parameters
    ----------
    json_file : str
        file path of QuPath exported GeoJSON
    Returns
    -------
    gj_data : dict
        dict of GeoJSON information
    shapes_np : dict
        dict of GeoJSON information stored in np.ndarray
            "array": np.ndarray - x,y point data in array
            "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
            "shape_name": str - name inherited from QuPath GeoJSON
    """
    if Path(json_file).suffix != ".zip":
        gj_data = json.load(open(json_file, "r"))
    else:
        with zipfile.ZipFile(json_file, "r") as z:
            for filename in z.namelist():
                with z.open(filename) as f:
                    data = f.read()
                    gj_data = json.loads(data.decode("utf-8"))

    shapes_np = [gj_to_np(s) for s in gj_data]
    gj_data = [add_unamed(gj) for gj in gj_data]
    return gj_data, shapes_np


def np_to_geojson(
    np_array: np.ndarray, shape_type="polygon", shape_name="unnamed"
):
    """convert np.ndarray to GeoJSON dict

    Parameters
    ----------
    np_array: np.ndarray
        coordinates of data
    shape_type:str
        GeoJSON shape type
    shape_name:str
        Name of the shape
    Returns
    -------
    shape_gj : dict
        dict of GeoJSON information
    shape_np : dict
        dict of GeoJSON information stored in np.ndarray
            "array": np.ndarray - x,y point data in array
            "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
            "shape_name": str - name inherited from QuPath GeoJSON
    """
    sh_type = shape_type.lower()

    gj_func = GJ_SHAPE_TYPE[sh_type]

    shape_gj = {
        "geometry": gj_func(np_array.tolist()),
        "properties": {"classification": {"name": shape_name}},
    }
    shape_np = {
        "array": np_array,
        "shape_type": shape_type.lower(),
        "shape_name": shape_name,
    }
    return shape_gj, shape_np


def shape_reader(shape_data, **kwargs):
    """
    Read shape data for transformation
    Shape data is stored as numpy arrays for operations but also as GeoJSON
    to contain metadata and interface with QuPath

    Parameters
    ----------
    shape_data: list of np.ndarray or str
        if str, will read data as GeoJSON file, if np.ndarray with assume
        it is coordinates
    kwargs
        keyword args passed to np_to_geojson convert

    Returns
    -------
    shapes_gj: list of dicts
        list of dicts of GeoJSON information
    shapes_np: list of dicts
        list of dicts of GeoJSON information stored in np.ndarray
        "array": np.ndarray - x,y point data in array
        "shape_type": str - indicates GeoJSON shape_type (Polygon, MultiPolygon, etc.)
        "shape_name": str - name inherited from QuPath GeoJSON
    """
    if isinstance(shape_data, list) is False:
        shape_data = [shape_data]

    shapes_gj = []
    shapes_np = []
    for sh in shape_data:
        if isinstance(sh, dict):
            out_shape_gj, out_shape_np = np_to_geojson(
                sh["array"], sh["shape_type"], sh["shape_name"]
            )

        elif isinstance(sh, np.ndarray):
            out_shape_gj, out_shape_np = np_to_geojson(sh, **kwargs)

        else:
            if Path(sh).is_file():
                sh_fp = Path(sh)

                if sh_fp.suffix in [".json", ".geojson", ".zip"]:
                    out_shape_gj, out_shape_np = read_geojson(str(sh_fp))
                # elif sh_fp.suffix == ".cz":
                #     out_shape_gj = read_zen_shapes(str(sh_fp))
                #     out_shape_np = [gj_to_np(s) for s in out_shape_gj]
                else:
                    raise ValueError(
                        "{} is not a geojson or numpy array".format(str(sh_fp))
                    )
            else:
                raise FileNotFoundError(
                    "{} file not found".format(str(sh_fp.as_posix()))
                )

        if isinstance(out_shape_gj, list):
            shapes_gj.extend(out_shape_gj)
        else:
            shapes_gj.append(out_shape_gj)

        if isinstance(out_shape_np, list):
            shapes_np.extend(out_shape_np)
        else:
            shapes_np.append(out_shape_np)

    return shapes_gj, shapes_np


def scale_shape_coordinates(poly: dict, scale_factor: float):
    """
    Scale coordinates by a factor

    Parameters
    ----------
    poly: dict
        dict of coordinate data contain np.ndarray in "array" key
    scale_factor: float
        isotropic scaling factor for the coordinates

    Returns
    -------
    poly: dict
        dict containing coordinates scaled by scale_factor
    """
    poly_coords = poly["array"]
    poly_coords = poly_coords * scale_factor
    poly["array"] = poly_coords
    return poly


def invert_nonrigid_transforms(itk_transforms: list):
    """
    Check list of sequential ITK transforms for non-linear (i.e., bspline) transforms
    Transformations need to be inverted to transform from moving to fixed space as transformations
    are mapped from fixed space to moving.
    This will first convert any non-linear transforms to a displacement field then invert the displacement field
    using ITK methods. It usually works quite well but is not an exact solution.
    Linear transforms can be inverted on  the fly when transforming points

    Parameters
    ----------
    itk_transforms:list
        list of itk.Transform

    Returns
    -------
    itk_transforms:list
        list of itk.Transform where any non-linear transforms are replaced with an inverted displacement field
    """
    tform_linear = [t.is_linear for t in itk_transforms]

    if all(tform_linear):
        return itk_transforms
    else:
        nl_idxs = np.where(np.array(tform_linear) == 0)[0]
        for nl_idx in nl_idxs:
            if not itk_transforms[nl_idx].inverse_transform:
                print(
                    f"transform at index {nl_idx} is non-linear and the inverse has not been computed\n"
                    "inverting displacement field(s)...\n"
                    "this can take some time"
                )
                itk_transforms[nl_idx].compute_inverse_nonlinear()

    return itk_transforms


def prepare_pt_transformation_data(transformations, compute_inverse=True):
    """
    Read and prepare wsireg transformation data for point set transformation

    Parameters
    ----------
    transformations
        list of dict containing elastix transformation data or str to wsireg .json file containing
        elastix transformation data
    compute_inverse : bool
        whether or not to compute the inverse transformation for moving to fixed point transformations
    Returns
    -------
    itk_pt_transforms:list
        list of transformation data ready to operate on points
    target_res:
        physical spacing of the final transformation in the transform sequence
        This is needed to map coordinates defined as pixel indices to physical coordinates and then back
    """
    if all([isinstance(t, RegTransform) for t in transformations]) is False:
        _, transformations = wsireg_transforms_to_itk_composite(
            transformations
        )
    if compute_inverse:
        transformations = invert_nonrigid_transforms(transformations)
    target_res = float(transformations[-1].output_spacing[0])
    return transformations, target_res


def itk_transform_pts(
    pt_data: np.ndarray,
    itk_transforms: list,
    px_idx=True,
    source_res=1,
    output_idx=True,
    target_res=2,
):
    """
    Transforms x,y points stored in np.ndarray using list of ITK transforms
    All transforms are in physical coordinates, so all points must be converted to physical coordinates
    before transformation, but this function allows converting back to pixel indices after transformation

    Can intake points in physical coordinates is px_idx == False
    Can output points in physical coordinates if output_idx == False

    Parameters
    ----------
    pt_data : np.ndarray
        array where rows are points and columns are x,y
    itk_transforms: list
        list of ITK transforms, non-linear transforms should be inverted
    px_idx: bool
        whether points are specified in physical coordinates (i.e., microns) or
        in pixel indices
    source_res: float
        resolution of the image on which annotations were made
    output_idx: bool
        whether transformed points should be output in physical coordinates (i.e., microns) or
        in pixel indices
    target_res: float
        resolution of the final target image for conversion back to pixel indices

    Returns
    -------
    tformed_pts:np.ndarray
        transformed points array where rows are points and columns are x,y

    """
    tformed_pts = []
    for pt in pt_data:
        if px_idx is True:
            pt = pt * source_res
        for idx, t in enumerate(itk_transforms):
            if idx == 0:
                t_pt = t.inverse_transform.TransformPoint(pt)
            else:
                t_pt = t.inverse_transform.TransformPoint(t_pt)
        t_pt = np.array(t_pt)
        if output_idx is True:
            t_pt *= 1 / target_res
        tformed_pts.append(t_pt)

    return np.stack(tformed_pts)


def transform_shapes(
    shape_data: list,
    itk_transforms: list,
    px_idx=True,
    source_res=1,
    output_idx=True,
    target_res=2,
):
    """
    Convenience function to apply itk_transform_pts to a list of shape data

    Parameters
    ----------
    shape_data:
        list of arrays where rows are points and columns are x,y
    itk_transforms: list
        list of ITK transforms, non-linear transforms should be inverted
    px_idx: bool
        whether points are specified in physical coordinates (i.e., microns) or
        in pixel indices
    source_res: float
        resolution of the image on which annotations were made
    output_idx: bool
        whether transformed points should be output in physical coordinates (i.e., microns) or
        in pixel indices
    target_res: float
        resolution of the final target image for conversion back to pixel indices

    Returns
    -------
        transformed_shape_data:list
            list of transformed np.ndarray data where rows are points and columns are x,y
    """
    transformed_shape_data = []
    for sh in shape_data:
        t_ptset = deepcopy(sh)
        ptset = sh.get("array")
        t_pts = itk_transform_pts(
            ptset,
            itk_transforms,
            px_idx=px_idx,
            source_res=source_res,
            output_idx=output_idx,
            target_res=target_res,
        )
        t_ptset["array"] = t_pts
        transformed_shape_data.append(t_ptset)

    return transformed_shape_data


def insert_transformed_pts_gj(gj_data: list, np_data: list):
    """
    insert point data into a list of geojson data

    Parameters
    ----------
    shape_gj : dict
        dict of GeoJSON information
    shape_np : dict
        transformed point data in wsireg shape dict

    Returns
    -------
    shape_gj : dict
        dict of GeoJSON information with updated coordinate information
    """

    gj_data_t = deepcopy(gj_data)
    for sh, gj in zip(np_data, gj_data_t):
        shape_type = gj.get("geometry").get("type")

        if shape_type == "Polygon":
            gj.get("geometry").update({"coordinates": [sh["array"].tolist()]})
        elif shape_type == "Point":
            gj.get("geometry").update(
                {"coordinates": np.squeeze(sh["array"]).tolist()}
            )
        elif shape_type == "MultiPoint":
            gj.get("geometry").update({"coordinates": sh["array"].tolist()})
        elif shape_type == "LineString":
            gj.get("geometry").update({"coordinates": sh["array"].tolist()})

    return gj_data_t


def get_int_dtype(value: int):
    """
    Determine appropriate bit precision for indexed image

    Parameters
    ----------
    value:int
        number of shapes

    Returns
    -------
    dtype:np.dtype
        apppropriate data type for index mask
    """
    if value <= np.iinfo(np.uint8).max:
        return np.uint8
    if value <= np.iinfo(np.uint16).max:
        return np.uint16
    if value <= np.iinfo(np.uint32).max:
        return np.int32
    else:
        raise ValueError("Too many shapes")


def get_all_shape_coords(shapes: list):
    return np.vstack(
        [np.squeeze(sh["geometry"]["coordinates"][0]) for sh in shapes]
    )


# code below is for managing transforms as masks rather than point sets
# will probably not reimplement, if segmentation data can is expressed
# as a mask, it can be transformed as an image (using attachment_modality)
def approx_polygon_contour(mask: np.ndarray, percent_arc_length=0.01):
    """
    Approximate binary mask contours to polygon vertices using cv2.

    Parameters
    ----------
    mask : numpy.ndarray
        2-d numpy array of datatype np.uint8.
    percent_arc_length : float
        scaling of epsilon for polygon approximate vertices accuracy.
        maximum distance of new vertices from original.

    Returns
    -------
    numpy.ndarray
        returns an 2d array of vertices, rows: points, columns: y,x

    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) > 1:
        contours = [contours[np.argmax([cnt.shape[0] for cnt in contours])]]

    epsilon = percent_arc_length * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    return np.squeeze(approx).astype(np.uint32)


def index_mask_to_shapes(index_mask, shape_name, tf_shapes):
    """
    Find the polygons of a transformed shape mask, conveting binary mask
    to list of polygon verteces and sorting by numerical index

    Parameters
    ----------
    index_mask:np.ndarray
        mask where each shape is defined by it's index
    shape_name:str
        name of the shape
    tf_shapes:list
        original list of shape GeoJSON data to be updated

    Returns
    -------
    updated_shapes:list
        dict of GeoJSON information with updated coordinate information
    """
    labstats = sitk.LabelShapeStatisticsImageFilter()
    labstats.SetBackgroundValue(0)
    labstats.Execute(index_mask)

    index_mask = sitk.GetArrayFromImage(index_mask)
    updated_shapes = deepcopy(tf_shapes)

    for idx, shape in enumerate(tf_shapes):
        if shape["properties"]["classification"]["name"] == shape_name:
            label_bb = labstats.GetBoundingBox(idx + 1)
            x_min = label_bb[0]
            x_len = label_bb[2]
            y_min = label_bb[1]
            y_len = label_bb[3]

            sub_mask = index_mask[y_min : y_min + y_len, x_min : x_min + x_len]

            sub_mask[sub_mask == idx + 1] = 255

            yx_coords = approx_polygon_contour(sub_mask, 0.00001)
            xy_coords = yx_coords
            xy_coords = np.append(xy_coords, xy_coords[:1, :], axis=0)
            xy_coords = xy_coords + [x_min, y_min]
            updated_shapes[idx]["geometry"]["coordinates"] = [
                xy_coords.tolist()
            ]

    return updated_shapes


# don't intend to maintain
# def read_zen_shapes(zen_fp):
#     """Read Zeiss Zen Blue .cz ROIs files to wsimap shapely format.
#
#     Parameters
#     ----------
#     zen_fp : str
#         file path of Zen .cz.
#
#     Returns
#     -------
#     list
#         list of wsimap shapely rois
#
#     """
#
#     root = etree.parse(zen_fp)
#
#     rois = root.xpath("//Elements")[0]
#     shapes_out = []
#     for shape in rois:
#         try:
#             ptset_name = shape.find("Attributes/Name").text
#         except AttributeError:
#             ptset_name = "unnamed"
#
#         if shape.tag == "Polygon":
#             ptset_cz = shape.find("Geometry/Points")
#             # ptset_type = "Polygon"
#
#             poly_str = ptset_cz.text
#             poly_str = poly_str.split(" ")
#             poly_str = [poly.split(",") for poly in poly_str]
#             poly_str = [[float(pt[0]), float(pt[1])] for pt in poly_str]
#
#             poly = {
#                 "geometry": geojson.Polygon(poly_str),
#                 "properties": {"classification": {"name": ptset_name}},
#             }
#
#             shapes_out.append(poly)
#
#         if shape.tag == "Rectangle":
#             rect_pts = shape.find("Geometry")
#
#             x = float(rect_pts.findtext("Left"))
#             y = float(rect_pts.findtext("Top"))
#             width = float(rect_pts.findtext("Width"))
#             height = float(rect_pts.findtext("Height"))
#
#             rect = geojson.Polygon(
#                 [
#                     [x, y],
#                     [x + width, y],
#                     [x + width, y + height],
#                     [x, y + height],
#                     [x, y],
#                 ]
#             )
#
#             rect = {
#                 "geometry": rect,
#                 "properties": {"classification": {"name": ptset_name}},
#             }
#
#             shapes_out.append(rect)
#
#     return shapes_out
