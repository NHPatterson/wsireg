from pathlib import Path
import json
import tempfile
import numpy as np
import zarr
import SimpleITK as sitk
from wsireg.parameter_maps.transformations import (
    BASE_RIG_TFORM,
    BASE_AFF_TFORM,
)
from wsireg.parameter_maps.reg_params import DEFAULT_REG_PARAM_MAPS
from wsireg.im_utils import read_image, std_prepro, contrast_enhance, sitk_inv_int

SITK_TO_NP_DTYPE = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.int64,
    7: np.uint64,
    8: np.float32,
    9: np.float64,
    10: np.complex64,
    11: np.complex64,
    12: np.int8,
    13: np.uint8,
    14: np.int16,
    15: np.int16,
    16: np.int32,
    17: np.int32,
    18: np.int64,
    19: np.int64,
    20: np.float32,
    21: np.float64,
    22: np.uint8,
    23: np.uint16,
    24: np.uint32,
    25: np.uint64,
}

COLNAME_TO_HEX = {
    "red": "FF0000",
    "green": "00FF00",
    "blue": "0000FF",
    "magenta": "FF00FF",
    "yellow": "FFFF00",
    "cyan": "00FFFFF",
    "white": "FFFFFF",
}


def sitk_pmap_to_dict(pmap):
    """
    convert SimpleElastix ParameterMap to python dictionary
    Parameters
    ----------
    pmap
        SimpleElastix ParameterMap

    Returns
    -------
    Python dict of SimpleElastix ParameterMap
    """
    pmap_dict = {}
    for k, v in pmap.items():
        if k in ["image", "invert"]:
            t_pmap = {}
            for k2, v2 in v.items():
                t_pmap[k2] = v2
            pmap_dict[k] = t_pmap
        else:
            pmap_dict[k] = v
    return pmap_dict


def pmap_dict_to_sitk(pmap_dict):
    """
    convert python dict to SimpleElastix ParameterMap
    Parameters
    ----------
    pmap_dict
        SimpleElastix ParameterMap in python dictionary

    Returns
    -------
    SimpleElastix ParameterMap of Python dict
    """
    pmap = sitk.ParameterMap()
    for k, v in pmap_dict.items():
        pmap[k] = v
    return pmap


def pmap_dict_to_json(pmap_dict, output_file):
    """
    save python dict of SimpleElastix to json
    Parameters
    ----------
    pmap_dict : dict
        parameter map stored in python dict
    output_file : str
        filepath of where to save the json
    """
    with open(output_file, "w") as fp:
        json.dump(pmap_dict, fp, indent=4)


def json_to_pmap_dict(json_file):
    """
    load python dict of SimpleElastix stored in json
    Parameters
    ----------
    json_file : dict
        filepath to json contained SimpleElastix parameter map
    """
    with open(json_file, "r") as fp:
        pmap_dict = json.load(fp)
    return pmap_dict


def prepare_tform_dict(tform_dict, shape_tform=False):

    tform_dict_out = {}
    for k, v in tform_dict.items():
        if k == "initial":
            tform_dict_out["initial"] = v
        else:
            tforms = []
            for tform in v:
                if "invert" in list(tform.keys()):
                    if shape_tform is False:
                        tforms.append(tform["image"])
                    else:
                        tforms.append(tform["invert"])
                else:
                    tforms.append(tform)
            tform_dict_out[k] = tforms

    return tform_dict_out


def parameter_load(reg_param):
    """Load a default registration parameter or one from file.

    Parameters
    ----------
    reg_param : str
        a string of the default parameterMap name [rigid, affine, nl]. If reg_model is not in the default list
        it should be a filepath to a SimpleITK parameterMap saved to .txt file on disk

    Returns
    -------
    sitk.ParameterMap


    """
    if isinstance(reg_param, str):
        if reg_param in list(DEFAULT_REG_PARAM_MAPS.keys()):
            reg_params = DEFAULT_REG_PARAM_MAPS[reg_param]
            reg_param_map = pmap_dict_to_sitk(reg_params)
        else:
            try:
                reg_param_map = sitk.ReadParameterFile(reg_param)
            except RuntimeError:
                print("invalid parameter file")

        return reg_param_map
    else:
        raise ValueError(
            "parameter input is not a filepath or default parameter file str"
        )


def register_2d_images(
    source_image,
    target_image,
    reg_params,
    reg_output_fp,
    histogram_match=False,
    compute_inverse=True,
    return_image=False,
):
    """
    Register 2D images with multiple models and return a list of elastix
    transformation maps.

    Parameters
    ----------
    source_image : SimpleITK.Image
        RegImage of image to be aligned
    target_image : SimpleITK.Image
        RegImage that is being aligned to (grammar is hard)
    reg_params : dict
        registration parameter maps stored in a dict, can be file paths to SimpleElastix parameterMaps stored
        as text or one of the default parameter maps (see parameter_load() function)
    reg_output_fp : str
        where to store registration outputs (iteration data and transformation files)
    histogram_match : bool
        whether to attempt histogram matching to improve registration
    compute_inverse : bool
        whether to compute the inverse for BSplineTransforms. This is needed to transform
        point sets
    Returns
    -------
    list of SimpleElastix transformation parameter maps

    """

    try:
        selx = sitk.SimpleElastix()
    except AttributeError:
        selx = sitk.ElastixImageFilter()

    selx.SetOutputDirectory(str(reg_output_fp))

    # these parameters may be made optional later
    # not critical though
    if histogram_match is True:
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(64)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        source_image.image = matcher.Execute(
            source_image.image, target_image.image
        )

    selx.SetMovingImage(source_image.image)
    selx.SetFixedImage(target_image.image)

    if source_image.mask is not None:
        selx.SetMovingMask(source_image.mask)

    if target_image.mask is not None:
        selx.SetFixedMask(target_image.mask)

    for idx, reg_param in enumerate(reg_params):
        if idx == 0:
            pmap = parameter_load(reg_param)
            pmap["WriteResultImage"] = ("false",)
            if target_image.mask is not None:
                pmap["AutomaticTransformInitialization"] = ("false",)
            selx.SetParameterMap(pmap)
        else:
            pmap = parameter_load(reg_param)
            pmap["WriteResultImage"] = ("false",)
            selx.AddParameterMap(pmap)

    selx.LogToConsoleOn()
    selx.LogToFileOn()

    # execute registration:
    if return_image is False:
        selx.Execute()
    else:
        image = selx.Execute()

    tform_list = list(selx.GetTransformParameterMap())

    if compute_inverse is True:
        compute_tforms = []
        for idx, reg_param in enumerate(reg_params):
            pmap = parameter_load(reg_param)
            pmap["WriteResultImage"] = ("false",)
            if pmap["Transform"][0] == "BSplineTransform":
                compute_tforms.append((idx, pmap))

        if len(compute_tforms) > 0:
            for idx, compute_tform in compute_tforms:

                selx.SetMovingImage(target_image.image)
                selx.SetFixedImage(target_image.image)
                compute_tform["Metric"] = ["DisplacementMagnitudePenalty"]
                max_step_double = compute_tform["MaximumStepLength"]
                compute_tform["MaximumStepLength"] = [
                    str(float(step) * 2) for step in max_step_double
                ]
                selx.SetParameterMap(compute_tform)

                with tempfile.TemporaryDirectory() as tempdir:
                    temp_tform_path = Path(tempdir) / "temp_tform.txt"
                    tform_out = list(selx.GetTransformParameterMap())[-1]
                    tform_out["InitialTransformParametersFileName"] = [
                        "NoInitialTransform"
                    ]
                    sitk.WriteParameterFile(
                        tform_out, str(temp_tform_path),
                    )
                    selx.SetInitialTransformParameterFileName(
                        str(temp_tform_path)
                    )
                    selx.SetOutputDirectory(tempdir)
                    selx.Execute()
                    inverted_tform = list(selx.GetTransformParameterMap())[0]

                tform_normal = tform_list[idx]
                tform_list[idx] = {
                    "image": tform_normal,
                    "invert": inverted_tform,
                }
        else:
            raise ValueError(
                "No support for inverting intermediate BSplineTransforms"
            )
    else:
        print("no inversions to compute")

    if return_image is False:
        return tform_list
    else:
        image = sitk.Cast(image, source_image.image.GetPixelID())
        return tform_list, image


def transform_2d_image(
    image, transformation_maps, writer="sitk", **zarr_kwargs
):
    """
    Transform 2D images with multiple models and return the transformed image
    or write the transformed image to disk as a .tif file.
    Multichannel or multicomponent images (RGB) have to be transformed a single channel at a time
    This function takes care of performing those transformations and reconstructing the image in the same
    data type as the input
    Parameters
    ----------
    image : SimpleITK.Image
        Image to be transformed
    transformation_maps : list
        list of SimpleElastix ParameterMaps to used for transformation
    Returns
    -------
    Transformed SimpleITK.Image
    """
    if transformation_maps is not None:

        try:
            tfx = sitk.TransformixImageFilter()
        except AttributeError:
            tfx = sitk.SimpleTransformix()

        # TODO: add mask cropping here later

        #     print("mask cropping")
        #     tmap = sitk.ReadParameterFile(transformation_maps[0])
        #     x_min = int(float(tmap["MinimumX"][0]))
        #     x_max = int(float(tmap["MaximumX"][0]))
        #     y_min = int(float(tmap["MinimumY"][0]))
        #     y_max = int(float(tmap["MaximumY"][0]))
        #     image = image[x_min:x_max, y_min:y_max]
        #     origin = np.repeat(0, len(image.GetSize()))
        #     image.SetOrigin(tuple([int(i) for i in origin]))

        # else:
        for idx, tmap in enumerate(transformation_maps):
            if isinstance(tmap, str):
                tmap = sitk.ReadParameterFile(tmap)

            if idx == 0:
                tmap["InitialTransformParametersFileName"] = (
                    "NoInitialTransform",
                )
                tfx.SetTransformParameterMap(tmap)
            else:
                tmap["InitialTransformParametersFileName"] = (
                    "NoInitialTransform",
                )

                tfx.AddTransformParameterMap(tmap)
        tfx.LogToConsoleOn()
        tfx.LogToFileOff()
    else:
        tfx = None

    if tfx is None:
        xy_final_size = np.array(image.GetSize(), dtype=np.uint32)
    else:
        xy_final_size = np.array(
            transformation_maps[-1]["Size"], dtype=np.uint32
        )

    if writer == "sitk" or writer is None:
        return transform_image_to_sitk(image, tfx)
    elif writer == "zarr":
        return transform_image_to_ome_zarr(
            image,
            tfx,
            xy_final_size,
            zarr_store_dir=zarr_kwargs["zarr_store_dir"],
            channel_names=zarr_kwargs["channel_names"],
            channel_colors=zarr_kwargs["channel_colors"],
        )
    else:
        raise ValueError("writer type {} not recognized".format(writer))


def transform_image_to_sitk(image, tfx):

    # manage transformation/casting if data is multichannel or RGB
    # data is always returned in the same PixelIDType as it is entered

    pixel_id = image.GetPixelID()
    if tfx is not None:
        if pixel_id in list(range(1, 13)) and image.GetDepth() == 0:
            tfx.SetMovingImage(image)
            image = tfx.Execute()
            image = sitk.Cast(image, pixel_id)

        elif pixel_id in list(range(1, 13)) and image.GetDepth() > 0:
            images = []
            for chan in range(image.GetDepth()):
                tfx.SetMovingImage(image[:, :, chan])
                images.append(sitk.Cast(tfx.Execute(), pixel_id))
            image = sitk.JoinSeries(images)
            image = sitk.Cast(image, pixel_id)

        elif pixel_id > 12:
            images = []
            for idx in range(image.GetNumberOfComponentsPerPixel()):
                im = sitk.VectorIndexSelectionCast(image, idx)
                pixel_id_nonvec = im.GetPixelID()
                tfx.SetMovingImage(im)
                images.append(sitk.Cast(tfx.Execute(), pixel_id_nonvec))
                del im

            image = sitk.Compose(images)
            image = sitk.Cast(image, pixel_id)

    return image


def calc_pyramid_levels(xy_final_shape):

    res_shape = xy_final_shape[::-1]
    res_shapes = [tuple(res_shape)]

    while all(res_shape > 256):
        res_shape = res_shape // 2
        res_shapes.append(tuple(res_shape))

    return res_shapes[:-1]


def add_ome_axes_single_plane(image_np):
    return image_np.reshape((1,) * (3) + image_np.shape)


def generate_channels(channel_names, channel_colors, pixel_id_np):
    channel_info = []
    for channel_name, channel_color in zip(channel_names, channel_colors):
        channel_info.append(
            {
                "label": channel_name,
                "color": channel_color,
                "active": True,
                "window": {"start": 0, "end": int(np.iinfo(pixel_id_np).max)},
            }
        )
    return channel_info


def transform_image_to_ome_zarr(
    image,
    tfx,
    xy_final_size,
    zarr_store_dir="myzarr",
    channel_names=None,
    channel_colors=None,
):

    pixel_id = image.GetPixelID()
    pixel_id_np = SITK_TO_NP_DTYPE.get(pixel_id)
    zarr_dtype = "{}{}".format(
        np.dtype(pixel_id_np).kind, np.dtype(pixel_id_np).itemsize
    )

    store = zarr.DirectoryStore(zarr_store_dir)
    grp = zarr.group(store, overwrite=True)
    pyr_levels = calc_pyramid_levels(xy_final_size[:2])

    if pixel_id < 12 and image.GetDepth() == 0:
        n_ch = 1
    else:
        n_ch = int(image.GetDepth())

    if pixel_id >= 13:
        n_ch = int(image.GetNumberOfComponentsPerPixel())

    pyr_shapes = [(1, n_ch, 1, int(pl[0]), int(pl[1])) for pl in pyr_levels]

    paths = []
    for path, pyr_shape in enumerate(pyr_levels):
        grp.create_dataset(
            str(path),
            shape=pyr_shapes[path],
            dtype=zarr_dtype,
            chunks=(1, 1, 1, 512, 512),
        )
        paths.append({"path": str(path)})

    multiscales = [{"version": "0.1", "datasets": paths,}]
    grp.attrs["multiscales"] = multiscales

    if pixel_id in list(range(1, 13)) and image.GetDepth() == 0:
        if tfx is not None:
            tfx.SetMovingImage(image)
            tformed_im = tfx.Execute()
            tformed_im = sitk.Cast(tformed_im, pixel_id)
        else:
            tformed_im = image

        for idx, path in enumerate(paths):
            if idx > 0:
                tformed_im = sitk.Shrink(tformed_im, (2, 2))

            grp[str(idx)][:] = add_ome_axes_single_plane(
                sitk.GetArrayFromImage(tformed_im)
            )
    elif pixel_id in list(range(1, 13)) and image.GetDepth() > 0:
        for chan in range(image.GetDepth()):
            if tfx is not None:
                tfx.SetMovingImage(image[:, :, chan])
                tformed_im = sitk.Cast(tfx.Execute(), pixel_id)
            else:
                tformed_im = image[:, :, chan]

            for idx, path in enumerate(paths):
                if idx > 0:
                    tformed_im = sitk.Shrink(tformed_im, (2, 2))

                grp[str(idx)][
                    :, chan : chan + 1, :, :, :
                ] = add_ome_axes_single_plane(
                    sitk.GetArrayFromImage(tformed_im)
                )
            del tformed_im

    elif pixel_id > 12:
        for vec_idx in range(image.GetNumberOfComponentsPerPixel()):
            tformed_im = sitk.VectorIndexSelectionCast(image, vec_idx)
            pixel_id_nonvec = tformed_im.GetPixelID()
            if tfx is not None:
                tfx.SetMovingImage(tformed_im)
                tformed_im = sitk.Cast(tfx.Execute(), pixel_id_nonvec)

            for idx, path in enumerate(paths):
                if idx > 0:
                    tformed_im = sitk.Shrink(tformed_im, (2, 2))

                grp[str(idx)][
                    :, vec_idx : vec_idx + 1, :, :, :
                ] = add_ome_axes_single_plane(
                    sitk.GetArrayFromImage(tformed_im)
                )
            del tformed_im

    if channel_names is None or n_ch != len(channel_names):
        channel_names = ["C{}".format(idx) for idx in range(n_ch)]

    n_colors = n_ch // len(COLNAME_TO_HEX) + 1
    color_palette = [*COLNAME_TO_HEX] * n_colors

    if channel_colors is None:
        channel_colors = [color_palette[idx] for idx in range(n_ch)]
    elif n_ch != len(channel_colors) and n_ch != 1:
        channel_colors = [color_palette[idx] for idx in range(n_ch)]
    elif n_ch != len(channel_colors) and n_ch == 1:
        channel_colors = ["FFFFFF"]
    else:
        channel_colors = [COLNAME_TO_HEX[ch] for ch in channel_colors]

    channel_info = generate_channels(
        channel_names, channel_colors, pixel_id_np
    )

    image_data = {
        'id': 1,
        'channels': channel_info,
        'rdefs': {'model': 'color',},
    }

    grp.attrs["omero"] = image_data

    return zarr_store_dir


def apply_transform_dict(
    image_fp,
    image_res,
    tform_dict_in,
    prepro_dict=None,
    is_shape_mask=False,
    writer="sitk",
    **im_tform_kwargs,
):
    """
    apply a complex series of transformations in a python dictionary to an image
    Parameters
    ----------
    image_fp : str
        file path to the image to be transformed, it will be read in it's entirety
    image_res : float
        pixel resolution of image to be transformed
    tform_dict : dict of lists
        dict of SimpleElastix transformations stored in lists, may contain an "initial" transforms (preprocessing transforms)
        these will be applied first, then the key order of the dict will determine the rest of the transformations
    prepro_dict : dict
        preprocessing to perform on image before transformation, default None reads full image
    is_shape_mask : bool
        whether the image being transformed is a shape mask (determines import)
    Returns
    -------
    SimpleITK.Image that has been transformed
    """

    if is_shape_mask is False:
        if isinstance(image_fp, sitk.Image):
            image = image_fp
        else:
            image = RegImage(
                image_fp, image_res, prepro_dict=prepro_dict
            ).image
    else:
        image = sitk.GetImageFromArray(image_fp)
        del image_fp
        image.SetSpacing((image_res, image_res))

    if tform_dict_in is None:
        if writer == "zarr":
            image = transform_2d_image(
                image,
                None,
                writer="zarr",
                zarr_store_dir=im_tform_kwargs["zarr_store_dir"],
                channel_names=im_tform_kwargs["channel_names"],
                channel_colors=im_tform_kwargs["channel_colors"],
            )
        else:
            image = transform_2d_image(image, None)

    else:
        tform_dict = tform_dict_in.copy()

        if tform_dict.get("registered") is None and tform_dict.get(0) is None:
            tform_dict["registered"] = tform_dict["initial"]
            tform_dict.pop("initial", None)

            if isinstance(tform_dict.get("registered"), list) is False:
                tform_dict["registered"] = [tform_dict["registered"]]

            for idx in range(len(tform_dict["registered"])):
                tform_dict[idx] = [tform_dict["registered"][idx]]

            tform_dict.pop("registered", None)
        else:
            tform_dict = prepare_tform_dict(tform_dict, shape_tform=False)

        if "initial" in tform_dict:
            for initial_tform in tform_dict["initial"]:
                if isinstance(initial_tform, list) is False:
                    initial_tform = [initial_tform]

                for tform in initial_tform:
                    image = transform_2d_image(image, [tform])

            tform_dict.pop("initial", None)

        for k, v in tform_dict.items():
            if writer == "zarr" and k == list(tform_dict.keys())[-1]:
                image = transform_2d_image(
                    image,
                    v,
                    writer="zarr",
                    zarr_store_dir=im_tform_kwargs["zarr_store_dir"],
                    channel_names=im_tform_kwargs["channel_names"],
                    channel_colors=im_tform_kwargs["channel_colors"],
                )
            else:
                print(v)
                image = transform_2d_image(image, v)

    return image


def apply_transform_dict_mod(
    image_fp,
    image_res,
    tform_dict_in,
    prepro_dict=None,
    is_shape_mask=False,
    writer="sitk",
    **im_tform_kwargs,
):
    """
    apply a complex series of transformations in a python dictionary to an image
    Parameters
    ----------
    image_fp : str
        file path to the image to be transformed, it will be read in it's entirety
    image_res : float
        pixel resolution of image to be transformed
    tform_dict : dict of lists
        dict of SimpleElastix transformations stored in lists, may contain an "initial" transforms (preprocessing transforms)
        these will be applied first, then the key order of the dict will determine the rest of the transformations
    prepro_dict : dict
        preprocessing to perform on image before transformation, default None reads full image
    is_shape_mask : bool
        whether the image being transformed is a shape mask (determines import)
    Returns
    -------
    SimpleITK.Image that has been transformed
    """

    if is_shape_mask is False:
        if isinstance(image_fp, sitk.Image):
            image = image_fp
        else:
            image = RegImage(
                image_fp, image_res, prepro_dict=prepro_dict
            ).image
    else:
        image = sitk.GetImageFromArray(image_fp)
        del image_fp
        image.SetSpacing((image_res, image_res))

    if tform_dict_in is None:
        if writer == "zarr":
            image = transform_2d_image(
                image,
                None,
                writer="zarr",
                zarr_store_dir=im_tform_kwargs["zarr_store_dir"],
                channel_names=im_tform_kwargs["channel_names"],
                channel_colors=im_tform_kwargs["channel_colors"],
            )
        else:
            image = transform_2d_image(image, None)

    else:

        tform_dict["registered"] = tform_dict.pop("registration")

        if tform_dict.get("registered") is None and tform_dict.get(0) is None:
            tform_dict["registered"] = tform_dict["initial"]
            tform_dict.pop("initial", None)

            if isinstance(tform_dict.get("registered"), list) is False:
                tform_dict["registered"] = [tform_dict["registered"]]

            for idx in range(len(tform_dict["registered"])):
                tform_dict[idx] = [tform_dict["registered"][idx]]

            tform_dict.pop("registered", None)
        else:
            tform_dict = prepare_tform_dict(tform_dict, shape_tform=False)

        if tform_dict.get("initial") is not None:
            for initial_tform in tform_dict.get("initial"):
                if isinstance(initial_tform, list) is False:
                    initial_tform = [initial_tform]

                for tform in initial_tform:
                    image = transform_2d_image(image, [tform])

        tform_dict.pop("initial", None)

        for k, v in tform_dict.items():

            if writer == "zarr" and k == list(tform_dict.keys())[-1]:
                image = transform_2d_image(
                    image,
                    v,
                    writer="zarr",
                    zarr_store_dir=im_tform_kwargs["zarr_store_dir"],
                    channel_names=im_tform_kwargs["channel_names"],
                    channel_colors=im_tform_kwargs["channel_colors"],
                )
            else:
                print(v)
                image = transform_2d_image(image, v)

    return image


def compute_rot_bound(image, angle=30):
    """
    compute the bounds of an image after by an angle

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated    angle
    angle : float
        angle of rotation in degrees, rotates counter-clockwise if positive

    Returns
    -------
    tuple of the rotated image's size in x and y

    """
    w, h = image.GetSize()[0], image.GetSize()[1]

    theta = np.radians(angle)
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    bound_w = (h * s) + (w * c)
    bound_h = (h * c) + (w * s)

    return bound_w, bound_h


def gen_rigid_tform_rot(image, spacing, angle):
    """
    generate a SimpleElastix transformation parameter Map to rotate image by angle
    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated
    spacing : float
        Physical spacing of the SimpleITK image
    angle : float
        angle of rotation in degrees, rotates counter-clockwise if positive

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    tform = BASE_RIG_TFORM.copy()
    image.SetSpacing((spacing, spacing))
    bound_w, bound_h = compute_rot_bound(image, angle=angle)

    rot_cent_pt = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w - 1) / 2, (bound_h - 1) / 2)
    )

    c_x, c_y = (image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2
    c_x_phy, c_y_phy = image.TransformContinuousIndexToPhysicalPoint(
        (c_x, c_y)
    )
    t_x = rot_cent_pt[0] - c_x_phy
    t_y = rot_cent_pt[1] - c_y_phy

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(int(np.ceil(bound_w))), str(int(np.ceil(bound_h)))]
    tform["CenterOfRotationPoint"] = [str(rot_cent_pt[0]), str(rot_cent_pt[1])]
    tform["TransformParameters"] = [
        str(np.radians(angle)),
        str(-1 * t_x),
        str(-1 * t_y),
    ]

    return tform


def gen_aff_tform_flip(image, spacing, flip="h"):
    """
    generate a SimpleElastix transformation parameter Map to horizontally or vertically flip image

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated
    spacing : float
        Physical spacing of the SimpleITK image
    flip : str
        "h" or "v" for horizontal or vertical flipping, respectively

    Returns
    -------
    SimpleITK.ParameterMap of flipping transformation (AffineTransform)

    """
    tform = BASE_AFF_TFORM.copy()
    image.SetSpacing((spacing, spacing))
    bound_w, bound_h = compute_rot_bound(image, angle=0)
    rot_cent_pt = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w - 1) / 2, (bound_h - 1) / 2)
    )

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(int(bound_w)), str(int(bound_h))]

    tform["CenterOfRotationPoint"] = [str(rot_cent_pt[0]), str(rot_cent_pt[1])]
    if flip == "h":
        tform_params = ["-1", "0", "0", "1", "0", "0"]
    elif flip == "v":
        tform_params = ["1", "0", "0", "-1", "0", "0"]

    tform["TransformParameters"] = tform_params

    return tform


class RegImage:
    """
    Image container class for reading images and preparing them for registration.
    Does two forms of preprocessing:
    Intensity preprocessing, modifying intensity values for best registration
    Spatial preprocessing, rotating and flipping images prior to registration for best alignment

    After preprocessing, images destined for registration should always be a single channel 2D image

    Parameters
    ----------
    image_fp : str
        filepath to image to be processed
    spacing : float
        Physical spacing (xy resolution) of the image (units are assumed to be the same for each image in experiment but are not
        defined)
        Does not yet support isotropic resolution
    prepro_dict : dict
        preprocessing dict to apply to image
    """

    def __init__(
        self,
        image_fp,
        image_res,
        prepro_dict={
            "image_type": "FL",
            "ch_indices": None,
            "as_uint8": False,
        },
        transforms=None,
        mask=None,
    ):

        self.image_filepath = Path(image_fp)
        self.image_res = image_res
        self.preprocessing = std_prepro()
        self.transforms = []
        self.mask = mask

        if prepro_dict is None:
            self.image = read_image(self.image_filepath, preprocessing=None)

            if self.image.GetDepth() > 0:
                self.image.SetSpacing((self.image_res, self.image_res, 1))
            else:
                self.image.SetSpacing((self.image_res, self.image_res))

            if self.mask is not None:
                self.mask = read_image(self.mask, preprocessing=None)
                self.mask.SetSpacing((self.image_res, self.image_res))

        else:

            for k, v in prepro_dict.items():
                self.preprocessing[k] = v

            (
                self.image,
                spatial_preprocessing,
            ) = self.preprocess_reg_image_intensity(
                str(self.image_filepath), self.preprocessing
            )

            if self.mask is not None:
                self.mask = read_image(str(self.mask), preprocessing=None)
                self.mask.SetSpacing((self.image_res, self.image_res))

            if self.image.GetDepth() >= 1:
                raise ValueError(
                    "preprocessing did not result in a single image plane"
                )

            if len(spatial_preprocessing) > 0 or transforms is not None:
                self.preprocess_reg_image_spatial(
                    spatial_preprocessing, transforms
                )

        if self.image.GetDepth() > 0:
            self.image.SetOrigin((0, 0, 0))
        else:
            self.image.SetOrigin((0, 0))
        if self.mask is not None:
            self.mask.SetOrigin((0, 0))

    def preprocess_reg_image_intensity(self, image_fp, preprocessing):

        image = read_image(image_fp, preprocessing)

        # separate spatial preprocessing from intensity preprocessing
        spatial_preprocessing = {}
        for spat_key in ["mask_bbox", "rot_cc", "flip"]:
            if spat_key in preprocessing:
                spatial_preprocessing[spat_key] = preprocessing[spat_key]
                preprocessing.pop(spat_key, None)

        # remove read time preprocessing
        preprocessing.pop("ch_indices", None)
        preprocessing.pop("as_uint8", None)

        # type specific
        if preprocessing["image_type"] == "FL":
            preprocessing.pop("inv_int", None)
        elif preprocessing["image_type"] == "BF":
            preprocessing.pop("max_int_proj", None)
            preprocessing.pop("contrast_enhance", None)

        if preprocessing.get("contrast_enhance_opt") is True:
            preprocessing.update({"contrast_enhance":contrast_enhance})
        else:
            preprocessing.pop("contrast_enhance", None)

        if preprocessing.get("inv_int_opt") is True:
            preprocessing.update({"inv_int":sitk_inv_int})
        else:
            preprocessing.pop("inv_int", None)

        preprocessing.pop("contrast_enhance_opt", None)
        preprocessing.pop("inv_int_opt", None)

        # remove type and set downsample to last
        preprocessing.pop("image_type", None)

        spatial_preprocessing.update(
            {"downsample": preprocessing.get("downsample")}
        )
        preprocessing.pop("downsample", None)

        # iterate through intensity transformations preprocessing dictionary
        for k, v in preprocessing.items():
            if v is not None:
                print("performing preprocessing: ", k)
                image = v(image)

        image.SetSpacing((self.image_res, self.image_res))

        return image, spatial_preprocessing

    def preprocess_reg_image_spatial(
        self, spatial_preprocessing, imported_transforms=None
    ):
        # spatial preprocessing:
        # imported transforms -> Masking -> rotation -> flipping
        if imported_transforms is not None:
            self.image = apply_transform_dict(
                self.image, self.image_res, imported_transforms
            )
            if self.mask is not None:
                self.mask = apply_transform_dict(
                    self.mask, self.image_res, imported_transforms
                )
            self.transforms.append(imported_transforms)

        if "mask_bbox" in spatial_preprocessing:
            bbox = spatial_preprocessing["mask_bbox"]
            self.image = self.image[
                bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[3]
            ]
            if self.mask is not None:
                self.mask = self.mask[
                    bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[3]
                ]
            self.transforms.append({"mask_bbox": bbox})

        if spatial_preprocessing.get("rot_cc") is not None:
            rotangle = spatial_preprocessing["rot_cc"]
            if rotangle is not None and rotangle != 0:
                print(f"rotating counter-clockwise {rotangle}")
                # self.image.SetSpacing((self.image_res, self.image_res))
                rot_tform = gen_rigid_tform_rot(
                    self.image, self.image_res, rotangle
                )
                self.image = transform_2d_image(self.image, [rot_tform])

                if self.mask is not None:
                    self.mask = transform_2d_image(self.mask, [rot_tform])

                self.transforms.append(rot_tform)

        if spatial_preprocessing.get("flip") is not None:
            flip_direction = spatial_preprocessing["flip"]
            if flip_direction != "None" and flip_direction is not None:
                print(f"flipping image {flip_direction}")

                flip_tform = gen_aff_tform_flip(
                    self.image, self.image_res, flip_direction
                )
                # image.SetSpacing((self.image_res, self.image_res))
                self.image = transform_2d_image(self.image, [flip_tform])

                if self.mask is not None:
                    self.mask = transform_2d_image(self.mask, [flip_tform])

                self.transforms.append(flip_tform)

        downsampling = spatial_preprocessing.get("downsample")
        # downsample single plane preprocessing
        if downsampling is not None and downsampling > 1:
            print("performing downsampling by factor: {}".format(downsampling))
            self.image.SetSpacing((self.image_res, self.image_res))
            self.image = sitk.Shrink(self.image, (downsampling, downsampling))

            if self.mask is not None:

                self.mask.SetSpacing((self.image_res, self.image_res))
                self.mask = sitk.Shrink(
                    self.mask, (downsampling, downsampling)
                )
