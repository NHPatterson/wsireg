from pathlib import Path
import tempfile
import SimpleITK as sitk
import cv2
import numpy as np
from lxml import etree
import geojson
from wsireg.reg_utils import prepare_tform_dict, pmap_dict_to_sitk, apply_transform_dict

GJ_SHAPE_TYPE = {
    "polygon": geojson.Polygon,
    "multipolygon": geojson.MultiPolygon,
    "point": geojson.Point,
    "multipoint": geojson.MultiPoint,
    "multilinestring": geojson.MultiLineString,
    "linestring": geojson.LineString,
}


def read_zen_shapes(zen_fp):
    """Read Zeiss Zen Blue .cz ROIs files to wsimap shapely format.

    Parameters
    ----------
    zen_fp : str
        file path of Zen .cz.

    Returns
    -------
    list
        list of wsimap shapely rois

    """

    root = etree.parse(zen_fp)

    rois = root.xpath("//Elements")[0]
    shapes_out = []
    for shape in rois:
        try:
            ptset_name = shape.find("Attributes/Name").text
        except AttributeError:
            ptset_name = "unnamed"

        if shape.tag == "Polygon":
            ptset_cz = shape.find("Geometry/Points")
            ptset_type = "Polygon"

            poly_str = ptset_cz.text
            poly_str = poly_str.split(" ")
            poly_str = [poly.split(",") for poly in poly_str]
            poly_str = [[float(pt[0]), float(pt[1])] for pt in poly_str]

            poly = {
                "geometry": geojson.Polygon(poly_str),
                "properties": {"classification": {"name": ptset_name}},
            }

            shapes_out.append(poly)

        if shape.tag == "Rectangle":
            rect_pts = shape.find("Geometry")

            x = float(rect_pts.findtext("Left"))
            y = float(rect_pts.findtext("Top"))
            width = float(rect_pts.findtext("Width"))
            height = float(rect_pts.findtext("Height"))

            rect = geojson.Polygon(
                [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height],
                    [x, y],
                ]
            )

            rect = {
                "geometry": rect,
                "properties": {"classification": {"name": ptset_name}},
            }

            shapes_out.append(rect)

    return shapes_out


def read_geojson(json_file):
    """Read GeoJSON files (and some QuPath metadata).

    Parameters
    ----------
    json_file : str
        file path of QuPath exported GeoJSON
    Returns
    -------
    dict
        dict of geojson information

    """

    return geojson.load(open(json_file, "r"))


def np_to_geojson(np_array, shape_type="polygon", shape_name="unnamed"):
    sh_type = shape_type.lower()

    gj_func = GJ_SHAPE_TYPE[sh_type]

    shape = {
        "geometry": gj_func(np_array.tolist()),
        "properties": {"classification": {"name": shape_name}},
    }

    return shape


def shape_reader(shape_data, **kwargs):
    if isinstance(shape_data, list) is False:
        shape_data = [shape_data]

    shapes = []
    for sh in shape_data:
        if isinstance(sh, dict):

            out_shapes = np_to_geojson(
                sh["array"], sh["shape_type"], sh["shape_name"]
            )

        elif isinstance(sh, np.ndarray):
            out_shapes = np_to_geojson(sh, **kwargs)

        else:
            if Path(sh).is_file():
                sh_fp = Path(sh)

                if sh_fp.suffix == ".json":
                    out_shapes = read_geojson(str(sh_fp))
                elif sh_fp.suffix == ".cz":
                    out_shapes = read_zen_shapes(str(sh_fp))
                else:
                    raise ValueError(
                        "{} is not a geojson, Zeiss Zen shape file or numpy array".format(
                            str(sh_fp)
                        )
                    )
        if isinstance(out_shapes, list):
            shapes.extend(out_shapes)
        else:
            shapes.append(out_shapes)

    return shapes


def read_elx_pts(pt_fp, transformation):
    x_scaling = float(transformation["Spacing"][0])
    y_scaling = float(transformation["Spacing"][1])

    pts_f = open(pt_fp, "r")
    lines = pts_f.readlines()

    pts_out = []
    for line in lines:
        pt = line.split("\t")[5]
        pts = pt.strip("; OutputPoint = [ ").strip(" ]").split(" ")
        pts_out.append(
            [float(pts[0]) * (1 / x_scaling), float(pts[1]) * (1 / y_scaling)]
        )

    return pts_out


def invert_linear_transformation(transformation, is_initial=True):

    if transformation["Transform"] == ["BSplineTransform"]:
        return transformation

    if transformation["Transform"] == ["EulerTransform"]:
        tform_params = transformation["TransformParameters"]
        inv_params = [str(-1 * float(param)) for param in tform_params]
        w, h = tuple([int(coord) for coord in transformation["Size"]])
        theta = float(inv_params[0])
        c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
        bound_w = (h * s) + (w * c)
        bound_h = (h * c) + (w * s)

        if is_initial is True:
            if int(w) != int(bound_w):
                inv_params[1] = str(float(inv_params[1]) * -1)
                # inv_params[2] = str(float(inv_params[1])*-1)

    if transformation["Transform"] == ["AffineTransform"]:
        inv_params = transformation["TransformParameters"]

    transformation["TransformParameters"] = inv_params

    return transformation


def write_shape_to_elx_pts(shape, source_image_res, fp):
    pts_f = open(str(fp), "w")
    n_pts = len(shape["geometry"]["coordinates"][0])
    pts_f.write("point\n")
    pts_f.write(str(n_pts) + "\n")

    if isinstance(source_image_res, float):
        source_image_res = (source_image_res, source_image_res)

    for pt in shape["geometry"]["coordinates"][0]:
        pts_f.write(
            "{} {}\n".format(
                float(pt[0]) * float(source_image_res[0]),
                float(pt[1]) * float(source_image_res[1]),
            )
        )
    pts_f.close()


def transform_points(shape, source_image_res, transformation):

    with tempfile.TemporaryDirectory() as tempdir:
        fp = Path(tempdir) / "elx_pts.txt"

        write_shape_to_elx_pts(shape, source_image_res, fp)

        try:
            tfx = sitk.TransformixImageFilter()
        except AttributeError:
            tfx = sitk.SimpleTransformix()

        if isinstance(transformation, list):
            print('combining tforms')
            for idx, tform in enumerate(transformation):
                if idx == 0:
                    tfx.SetTransformParameterMap(tform)
                else:
                    tfx.AddTransformParameterMap(tform)

            transformation = transformation[-1]
        else:
            tfx.SetTransformParameterMap(transformation)

        tfx.SetFixedPointSetFileName(str(fp))
        tfx.SetOutputDirectory(tempdir)
        tfx.LogToFileOff()
        tfx.LogToConsoleOff()
        tfx.Execute()
        fp_out = Path(tempdir) / "outputpoints.txt"

        transformed_pts = read_elx_pts(str(fp_out), transformation)

        tf_shape = shape.copy()
        tf_shape["geometry"] = geojson.Polygon([transformed_pts])

        final_spacing = transformation["Spacing"]

    return tf_shape, final_spacing


#
#
# if "initial" in tform_dict:
#     for initial_tform in tform_dict["initial"]:
#         if isinstance(initial_tform, list):
#             initial_tform = initial_tform
#         else:
#             initial_tform = [initial_tform]
#         image = transform_2d_image(image, image_res, initial_tform)
#         tform_dict.pop("initial", None)
# for k, v in tform_dict.items():
#     image = transform_2d_image(image, image_res, v)
#
# return image


def create_flat_tform_list(tform_dict):
    tform_list = []
    tform_dict = prepare_tform_dict(tform_dict, shape_tform=True)
    for k, v in tform_dict.items():
        if k == "initial" and len(v) == 1:
            for tform in v[0]:
                tform["InitialTransformParametersFileName"] = [
                    "NoInitialTransform"
                ]
                tform = invert_linear_transformation(
                    tform.copy(), is_initial=True
                )

                tform_list.append(tform)
        else:
            for tform in v:
                tform["InitialTransformParametersFileName"] = [
                    "NoInitialTransform"
                ]
                if k == "initial":
                    tform = invert_linear_transformation(
                        tform.copy(), is_initial=True
                    )
                else:
                    tform = invert_linear_transformation(
                        tform.copy(), is_initial=False
                    )

                tform_list.append(tform)

    return tform_list


def apply_transformorm_dict_shapes(shape, source_image_res, tform_dict):
    tform_list = create_flat_tform_list(tform_dict)

    for idx, tf in enumerate(tform_list):
        print('applying:')
        print(tf)
        if idx == 0:
            shape, new_spacing = transform_points(shape, source_image_res, tf)
        else:
            shape, new_spacing = transform_points(shape, new_spacing, tf)

    return shape


def get_int_dtype(value):
    if value <= np.iinfo(np.uint8).max:
        return np.uint8
    if value <= np.iinfo(np.uint16).max:
        return np.uint16
    if value <= np.iinfo(np.uint32).max:
        return np.uint32
    else:
        raise ValueError("Too many shapes")


def get_all_shape_coords(shapes):
    return np.concatenate(
        np.asarray([sh["geometry"]["coordinates"][0] for sh in shapes]), axis=0
    )


def shapes_to_mask_dict(shapes):
    shape_names = np.unique(
        [sh["properties"]["classification"]["name"] for sh in shapes]
    )
    mat_dtype = get_int_dtype(len(shapes))

    all_coords = get_all_shape_coords(shapes)

    x_max = np.max(all_coords[:, 0])
    y_max = np.max(all_coords[:, 1])

    mask_dict = {}

    for shape_name in np.unique(shape_names):
        mask_dict[shape_name] = np.zeros((y_max, x_max), dtype=mat_dtype)

    for idx, shape in enumerate(shapes):
        shape_name = shape["properties"]["classification"]["name"]
        shape_coords = np.array(shape["geometry"]["coordinates"])

        mask_dict[shape_name] = cv2.fillPoly(
            mask_dict[shape_name], shape_coords, idx + 1
        )
    return mask_dict


def approx_polygon_contour(mask, percent_arc_length=0.01):
    """Approximate binary mask contours to polygon vertices using cv2.

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
    epsilon = percent_arc_length * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    return np.squeeze(approx).astype(np.uint32)


def index_mask_to_shapes(index_mask, shape_name, shapes):

    index_mask = sitk.GetArrayFromImage(index_mask)
    tf_shapes = shapes.copy()

    for idx, shape in enumerate(shapes):
        if shape["properties"]["classification"]["name"] == shape_name:
            sub_mask = index_mask
            sub_mask[sub_mask == idx + 1] = 255

            yx_coords = approx_polygon_contour(sub_mask, 0.001)
            xy_coords = yx_coords
            xy_coords = np.append(xy_coords, xy_coords[:1, :], axis=0)

            tf_shapes[idx]["geometry"]["coordinates"] = [xy_coords.tolist()]

    return tf_shapes


def apply_transformation_dict_shapes(shapes, image_res, tform_dict):
    mask_dict = shapes_to_mask_dict(shapes)

    tf_shapes = shapes.copy()

    for k,v in mask_dict.items():
        print("transforming shapes: {}".format(k))
        tformed_mask = apply_transform_dict(v, image_res, tform_dict, is_shape_mask=True)
        tf_shapes = index_mask_to_shapes(tformed_mask,k, tf_shapes)

    return tf_shapes
