from copy import deepcopy

import SimpleITK as sitk


def euler_elx_to_itk2d(tform):
    euler2d = sitk.Euler2DTransform()

    center = [float(p) for p in tform['CenterOfRotationPoint']]
    euler2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform['TransformParameters']]
    euler2d.SetParameters(elx_parameters)

    return euler2d


def similarity_elx_to_itk2d(tform):
    similarity2d = sitk.Similarity2DTransform()

    center = [float(p) for p in tform['CenterOfRotationPoint']]
    similarity2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform['TransformParameters']]
    similarity2d.SetParameters(elx_parameters)

    return similarity2d


def affine_elx_to_itk2d(tform):
    im_dimension = len(tform["Size"])
    affine2d = sitk.AffineTransform(im_dimension)

    center = [float(p) for p in tform['CenterOfRotationPoint']]
    affine2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform['TransformParameters']]
    affine2d.SetParameters(elx_parameters)

    return affine2d


def bspline_elx_to_itk2d(tform):
    im_dimension = len(tform["Size"])

    bspline2d = sitk.BSplineTransform(im_dimension, 3)
    bspline2d.SetTransformDomainOrigin(
        [float(p) for p in tform['Origin']]
    )  # from fixed image
    bspline2d.SetTransformDomainPhysicalDimensions(
        [int(p) for p in tform['Size']]
    )  # from fixed image
    bspline2d.SetTransformDomainDirection(
        [float(p) for p in tform['Direction']]
    )  # from fixed image

    fixedParams = [int(p) for p in tform['GridSize']]
    fixedParams += [float(p) for p in tform['GridOrigin']]
    fixedParams += [float(p) for p in tform['GridSpacing']]
    fixedParams += [float(p) for p in tform['GridDirection']]
    bspline2d.SetFixedParameters(fixedParams)
    bspline2d.SetParameters([float(p) for p in tform['TransformParameters']])
    return bspline2d


def convert_to_itk(tform):

    if tform["Transform"][0] == "AffineTransform":
        itk_tform = affine_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "SimilarityTransform":
        itk_tform = similarity_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "EulerTransform":
        itk_tform = euler_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "BSplineTransform":
        itk_tform = bspline_elx_to_itk2d(tform)

    itk_tform.OutputSpacing = [float(p) for p in tform["Spacing"]]
    itk_tform.OutputDirection = [float(p) for p in tform["Direction"]]
    itk_tform.OutputOrigin = [float(p) for p in tform["Origin"]]
    itk_tform.OutputSize = [int(p) for p in tform["Size"]]
    itk_tform.ResampleInterpolator = tform["ResampleInterpolator"][0]

    return itk_tform


def get_elastix_transforms(transformations):
    elastix_transforms = deepcopy(transformations)

    for k, v in elastix_transforms.items():
        elastix_transforms.update({k: [t.elastix_transform for t in v]})

    return elastix_transforms
