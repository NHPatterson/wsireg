from pathlib import Path
import json
import tempfile
import SimpleITK as sitk
from wsireg.parameter_maps.reg_params import DEFAULT_REG_PARAM_MAPS


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
                        tform_out,
                        str(temp_tform_path),
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
