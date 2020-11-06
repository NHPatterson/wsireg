import numpy as np
import SimpleITK as sitk
from wsireg.parameter_maps.transformations import (
    BASE_RIG_TFORM,
    BASE_AFF_TFORM,
)


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

    # if tfx is None:
    #     xy_final_size = np.array(image.GetSize(), dtype=np.uint32)
    # else:
    #     xy_final_size = np.array(
    #         transformation_maps[-1]["Size"], dtype=np.uint32
    #     )

    if writer == "sitk" or writer is None:
        return transform_image_to_sitk(image, tfx)
    elif writer == "zarr":
        return
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
        # else:
        #     image = RegImage(
        #         image_fp, image_res, prepro_dict=prepro_dict
        #     ).image
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
