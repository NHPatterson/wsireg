from enum import Enum, EnumMeta
from pathlib import Path

DEFAULT_REG_PARAM_MAPS = {
    'rigid': {
        "AutomaticScalesEstimation": ['true'],
        "AutomaticTransformInitialization": ['true'],
        "BSplineInterpolationOrder": ['1'],
        "CompressResultImage": ['true'],
        "DefaultPixelValue": ['0'],
        "ErodeMask": ['false'],
        "FinalBSplineInterpolationOrder": ['1'],
        "FixedImageDimension": ['2'],
        "FixedImagePyramid": ['FixedRecursiveImagePyramid'],
        "FixedInternalImagePixelType": ['float'],
        "HowToCombineTransforms": ['Compose'],
        "ImageSampler": ["Random"],
        "Interpolator": ['LinearInterpolator'],
        "MaximumNumberOfIterations": ['200'],
        "MaximumNumberOfSamplingAttempts": [
            '10',
        ],
        "MaximumStepLength": [
            '100.0',
            '75.0',
            '66.0',
            '50.0',
            '25.0',
            '15.0',
            '10.0',
            '10.0',
            '5.0',
            '1.0',
        ],
        "Metric": ['AdvancedMattesMutualInformation'],
        "MovingImageDimension": ['2'],
        "MovingImagePyramid": ['MovingRecursiveImagePyramid'],
        "MovingInternalImagePixelType": ['float'],
        "NewSamplesEveryIteration": ['true'],
        "NumberOfHistogramBins": ['16'],
        "NumberOfResolutions": ['10'],
        "NumberOfSpatialSamples": ['10000'],
        "Optimizer": ['AdaptiveStochasticGradientDescent'],
        "Registration": ['MultiResolutionRegistration'],
        "RequiredRatioOfValidSamples": ['0.05'],
        "ResampleInterpolator": ['FinalNearestNeighborInterpolator'],
        "Resampler": ['DefaultResampler'],
        "ResultImageFormat": ['mha'],
        "ResultImagePixelType": ['short'],
        "Transform": ['EulerTransform'],
        "UseDirectionCosines": ['true'],
        "WriteResultImage": ['false'],
        "WriteTransformParametersEachResolution": ['true'],
    },
    'affine': {
        "AutomaticScalesEstimation": ['true'],
        "AutomaticTransformInitialization": ['true'],
        "BSplineInterpolationOrder": ['1'],
        "CompressResultImage": ['true'],
        "DefaultPixelValue": ['0'],
        "ErodeMask": ['false'],
        "FinalBSplineInterpolationOrder": ['1'],
        "FixedImageDimension": ['2'],
        "FixedImagePyramid": ['FixedRecursiveImagePyramid'],
        "FixedInternalImagePixelType": ['float'],
        "HowToCombineTransforms": ['Compose'],
        "ImageSampler": ['Random'],
        "Interpolator": ['LinearInterpolator'],
        "MaximumNumberOfIterations": ['200'],
        "MaximumNumberOfSamplingAttempts": [
            '10',
        ],
        "MaximumStepLength": [
            '100.0',
            '75.0',
            '66.0',
            '50.0',
            '25.0',
            '15.0',
            '10.0',
            '10.0',
            '5.0',
            '1.0',
        ],
        "Metric": ['AdvancedMattesMutualInformation'],
        "MovingImageDimension": ['2'],
        "MovingImagePyramid": ['MovingRecursiveImagePyramid'],
        "MovingInternalImagePixelType": ['float'],
        "NewSamplesEveryIteration": ['true'],
        "NumberOfHistogramBins": ['32'],
        "NumberOfResolutions": ['10'],
        "NumberOfSpatialSamples": ['10000'],
        "Optimizer": ['AdaptiveStochasticGradientDescent'],
        "Registration": ['MultiResolutionRegistration'],
        "RequiredRatioOfValidSamples": ['0.05'],
        "ResampleInterpolator": ['FinalNearestNeighborInterpolator'],
        "Resampler": ['DefaultResampler'],
        "ResultImageFormat": ['mha'],
        "ResultImagePixelType": ['short'],
        "Transform": ['AffineTransform'],
        "UseDirectionCosines": ['true'],
        "WriteResultImage": ['false'],
        "WriteTransformParametersEachResolution": ['true'],
    },
    'similarity': {
        "AutomaticScalesEstimation": ['true'],
        "AutomaticTransformInitialization": ['true'],
        "BSplineInterpolationOrder": ['1'],
        "CompressResultImage": ['true'],
        "DefaultPixelValue": ['0'],
        "ErodeMask": ['false'],
        "FinalBSplineInterpolationOrder": ['1'],
        "FixedImageDimension": ['2'],
        "FixedImagePyramid": ['FixedRecursiveImagePyramid'],
        "FixedInternalImagePixelType": ['float'],
        "HowToCombineTransforms": ['Compose'],
        "ImageSampler": ['Random'],
        "Interpolator": ['LinearInterpolator'],
        "MaximumNumberOfIterations": ['200'],
        "MaximumNumberOfSamplingAttempts": [
            '10',
        ],
        "MaximumStepLength": [
            '100.0',
            '75.0',
            '66.0',
            '50.0',
            '25.0',
            '15.0',
            '10.0',
            '10.0',
            '5.0',
            '1.0',
        ],
        "Metric": ['AdvancedMattesMutualInformation'],
        "MovingImageDimension": ['2'],
        "MovingImagePyramid": ['MovingRecursiveImagePyramid'],
        "MovingInternalImagePixelType": ['float'],
        "NewSamplesEveryIteration": ['true'],
        "NumberOfHistogramBins": ['32'],
        "NumberOfResolutions": ['10'],
        "NumberOfSpatialSamples": ['10000'],
        "Optimizer": ['AdaptiveStochasticGradientDescent'],
        "Registration": ['MultiResolutionRegistration'],
        "RequiredRatioOfValidSamples": ['0.05'],
        "ResampleInterpolator": ['FinalNearestNeighborInterpolator'],
        "Resampler": ['DefaultResampler'],
        "ResultImageFormat": ['mha'],
        "ResultImagePixelType": ['short'],
        "Transform": ['SimilarityTransform'],
        "UseDirectionCosines": ['true'],
        "WriteResultImage": ['false'],
        "WriteTransformParametersEachResolution": ['true'],
    },
    'nl': {
        "AutomaticScalesEstimation": ['true'],
        "AutomaticTransformInitialization": ['false'],
        "BSplineInterpolationOrder": ['1'],
        "CompressResultImage": ['true'],
        "DefaultPixelValue": ['0'],
        "ErodeMask": ['false'],
        "FinalBSplineInterpolationOrder": ['1'],
        "FinalGridSpacingInPhysicalUnits": ['100'],
        "FixedImageDimension": ['2'],
        "FixedImagePyramid": ['FixedRecursiveImagePyramid'],
        "FixedInternalImagePixelType": ['float'],
        "GridSpacingSchedule": [
            '512',
            '512',
            '392',
            '392',
            '256',
            '256',
            '128',
            '128',
            '64',
            '64',
            '32',
            '32',
            '16',
            '16',
            '4',
            '4',
            '2',
            '2',
            '1',
            '1',
        ],
        "HowToCombineTransforms": ['Compose'],
        "ImageSampler": ['Random'],
        "Interpolator": ['LinearInterpolator'],
        "MaximumNumberOfIterations": ['200'],
        "MaximumNumberOfSamplingAttempts": [
            '10',
        ],
        "MaximumStepLength": [
            '100',
            '90',
            '70',
            '50',
            '40',
            '30',
            '20',
            '10',
            '1',
            '1',
        ],
        "Metric": ['AdvancedMattesMutualInformation'],
        "MovingImageDimension": ['2'],
        "MovingImagePyramid": ['MovingRecursiveImagePyramid'],
        "MovingInternalImagePixelType": ['float'],
        "NewSamplesEveryIteration": ['true'],
        "NumberOfHistogramBins": ['32'],
        "NumberOfResolutions": ['10'],
        "NumberOfSpatialSamples": ['50000'],
        "Optimizer": ['AdaptiveStochasticGradientDescent'],
        "Registration": ['MultiResolutionRegistration'],
        "RequiredRatioOfValidSamples": ['0.05'],
        "ResampleInterpolator": ['FinalNearestNeighborInterpolator'],
        "Resampler": ['DefaultResampler'],
        "ResultImageFormat": ['mha'],
        "ResultImagePixelType": ['short'],
        "Transform": ['BSplineTransform'],
        "UseDirectionCosines": ['true'],
        "WriteResultImage": ['false'],
        "WriteTransformParametersEachResolution": ['true'],
    },
    'fi_correction': {
        "AutomaticScalesEstimation": ['true'],
        "AutomaticTransformInitialization": ['false'],
        "BSplineInterpolationOrder": ['1'],
        "CompressResultImage": ['true'],
        "DefaultPixelValue": ['0'],
        "ErodeMask": ['false'],
        "FinalBSplineInterpolationOrder": ['1'],
        "FixedImageDimension": ['2'],
        "FixedImagePyramid": ['FixedRecursiveImagePyramid'],
        "FixedInternalImagePixelType": ['float'],
        "HowToCombineTransforms": ['Compose'],
        "ImagePyramidSchedule": ['8', '8', '4', '4', '2', '2', '1', '1'],
        "ImageSampler": ['Random'],
        "Interpolator": ['LinearInterpolator'],
        "MaximumNumberOfIterations": ['75'],
        "MaximumNumberOfSamplingAttempts": [
            '10',
        ],
        "MaximumStepLength": ['100', '50', '20', '10'],
        "Metric": ['AdvancedMattesMutualInformation'],
        "MovingImageDimension": ['2'],
        "MovingImagePyramid": ['MovingRecursiveImagePyramid'],
        "MovingInternalImagePixelType": ['float'],
        "NewSamplesEveryIteration": ['true'],
        "NumberOfHistogramBins": ['16'],
        "NumberOfResolutions": ['4'],
        "NumberOfSpatialSamples": ['10000'],
        "Optimizer": ['AdaptiveStochasticGradientDescent'],
        "Registration": ['MultiResolutionRegistration'],
        "RequiredRatioOfValidSamples": ['0.05'],
        "ResampleInterpolator": ['FinalNearestNeighborInterpolator'],
        "Resampler": ['DefaultResampler'],
        "ResultImageFormat": ['mha'],
        "ResultImagePixelType": ['short'],
        "Transform": ['EulerTransform'],
        "UseDirectionCosines": ['true'],
        "WriteResultImage": ['true'],
        "WriteTransformParametersEachResolution": ['false'],
    },
}

test_rig = DEFAULT_REG_PARAM_MAPS["rigid"].copy()
test_aff = DEFAULT_REG_PARAM_MAPS["affine"].copy()
test_sim = DEFAULT_REG_PARAM_MAPS["similarity"].copy()
test_nl = DEFAULT_REG_PARAM_MAPS["nl"].copy()

test_rig["MaximumNumberOfIterations"] = ["10"]
test_aff["MaximumNumberOfIterations"] = ["10"]
test_sim["MaximumNumberOfIterations"] = ["10"]
test_nl["MaximumNumberOfIterations"] = ["10"]

DEFAULT_REG_PARAM_MAPS["rigid_test"] = test_rig
DEFAULT_REG_PARAM_MAPS["affine_test"] = test_aff
DEFAULT_REG_PARAM_MAPS["similarity_test"] = test_sim
DEFAULT_REG_PARAM_MAPS["nl_test"] = test_nl


def elx_lineparser(line):
    if line[0] == "(":
        params = (
            line.replace("(", "")
            .replace(")", "")
            .replace("\n", "")
            .replace('"', "")
        )
        params = params.split(" ", 1)
        k, v = params[0], params[1]
        if " " in v:
            v = v.split(" ")
            v = list(filter(lambda a: a != "", v))
        if isinstance(v, list) is False:
            v = [v]
        return k, v
    else:
        return None, None


def read_elastix_parameter_file(elx_fp):
    with open(
        elx_fp,
        "r",
    ) as f:
        lines = f.readlines()
    parameters = {}
    for line in lines:
        k, v = elx_lineparser(line)
        if k is not None:
            parameters.update({k: v})
    return parameters


class RegParamsMeta(EnumMeta):
    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except (TypeError, KeyError):
            if isinstance(name, str) and Path(name).exists():
                return read_elastix_parameter_file(name)
            else:
                raise ValueError(
                    "unrecognized registration parameter, please provide"
                    "file path to elastix transform parameters or specify one of "
                    f"{[i.name for i in self]}"
                )


class RegParams(Enum, metaclass=RegParamsMeta):
    rigid = DEFAULT_REG_PARAM_MAPS["rigid"]
    similarity = DEFAULT_REG_PARAM_MAPS["similarity"]
    affine = DEFAULT_REG_PARAM_MAPS["affine"]
    nl = DEFAULT_REG_PARAM_MAPS["nl"]
    nonlinear = DEFAULT_REG_PARAM_MAPS["nl"]
    bspline = DEFAULT_REG_PARAM_MAPS["nl"]
    rigid_test = DEFAULT_REG_PARAM_MAPS["rigid_test"]
    affine_test = DEFAULT_REG_PARAM_MAPS["affine_test"]
    nl_test = DEFAULT_REG_PARAM_MAPS["nl_test"]
