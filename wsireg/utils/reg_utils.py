import json
import numpy as np
import SimpleITK as sitk
import itk
from wsireg.parameter_maps.reg_params import DEFAULT_REG_PARAM_MAPS
from wsireg.utils.itk_im_conversions import (
    itk_image_to_sitk_image,
)

NP_TO_SITK_DTYPE = {
    np.dtype(np.int8): 0,
    np.dtype(np.uint8): 1,
    np.dtype(np.int16): 2,
    np.dtype(np.uint16): 3,
    np.dtype(np.int32): 4,
    np.dtype(np.uint32): 5,
    np.dtype(np.int64): 6,
    np.dtype(np.uint64): 7,
    np.dtype(np.float32): 8,
    np.dtype(np.float64): 9,
    np.dtype(np.complex64): 10,
    np.dtype(np.complex64): 11,
}


def sitk_pmap_to_dict(pmap):
    """
    Convert SimpleElastix ParameterMap to python dictionary

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
    Convert python dict to SimpleElastix ParameterMap

    Parameters
    ----------
    pmap_dict
        SimpleElastix ParameterMap in python dictionary

    Returns
    -------
    SimpleElastix ParameterMap of Python dict
    """
    # pmap = sitk.ParameterMap()
    # pmap = {}
    # for k, v in pmap_dict.items():
    #     pmap[k] = v
    return pmap_dict


def pmap_dict_to_json(pmap_dict, output_file):
    """
    Save python dict of ITKElastix to json

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
    Load python dict of SimpleElastix stored in json

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


def parameter_to_itk_pobj(reg_param_map):
    """
    Transfer parameter data stored in dict to ITKElastix ParameterObject

    Parameters
    ----------
    reg_param_map: dict
        elastix registration parameters

    Returns
    -------
    itk_param_map:itk.ParameterObject
        ITKElastix object for registration parameters
    """
    parameter_object = itk.ParameterObject.New()
    itk_param_map = parameter_object.GetDefaultParameterMap("rigid")
    for k, v in reg_param_map.items():
        itk_param_map[k] = v
    return itk_param_map


def register_2d_images_itkelx(
    source_image,
    target_image,
    reg_params,
    reg_output_fp,
    histogram_match=False,
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
    Returns
    -------
        tform_list: list
            list of ITKElastix transformation parameter maps
        image: itk.Image
            resulting registered moving image
    """
    if histogram_match is True:
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(64)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        source_image.image = matcher.Execute(
            source_image.image, target_image.image
        )
    source_image.sitk_to_itk()
    target_image.sitk_to_itk()

    selx = itk.ElastixRegistrationMethod.New(
        source_image.image, target_image.image
    )

    # Set additional options
    selx.SetLogToConsole(True)
    selx.SetOutputDirectory(str(reg_output_fp))

    if source_image.mask is not None:
        selx.SetMovingMask(source_image.mask)

    if target_image.mask is not None:
        selx.SetFixedMask(target_image.mask)

    selx.SetMovingImage(source_image.image)
    selx.SetFixedImage(target_image.image)

    parameter_object_registration = itk.ParameterObject.New()
    for idx, reg_param in enumerate(reg_params):
        if idx == 0:
            reg_param["WriteResultImage"] = ("false",)
            if target_image.mask is not None:
                reg_param["AutomaticTransformInitialization"] = ("false",)
            else:
                reg_param["AutomaticTransformInitialization"] = ('true',)

            parameter_object_registration.AddParameterMap(reg_param)
        else:
            reg_param["WriteResultImage"] = ("false",)
            reg_param["AutomaticTransformInitialization"] = ('false',)
            parameter_object_registration.AddParameterMap(reg_param)

    selx.SetParameterObject(parameter_object_registration)

    # Update filter object (required)
    selx.UpdateLargestPossibleRegion()

    # Results of Registration
    result_transform_parameters = selx.GetTransformParameterObject()

    # execute registration:
    tform_list = []
    for idx in range(result_transform_parameters.GetNumberOfParameterMaps()):
        tform = {}
        for k, v in result_transform_parameters.GetParameterMap(idx).items():
            tform[k] = v
        tform_list.append(tform)

    if return_image is False:
        return tform_list
    else:
        image = selx.GetOutput()
        image = itk_image_to_sitk_image(image)
        pixel_id = NP_TO_SITK_DTYPE[source_image.im_dtype]
        image = sitk.Cast(image, pixel_id)
        return tform_list, image
