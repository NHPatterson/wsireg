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

# advanced mean squares
ams_rig = DEFAULT_REG_PARAM_MAPS["rigid"].copy()
ams_aff = DEFAULT_REG_PARAM_MAPS["affine"].copy()
ams_sim = DEFAULT_REG_PARAM_MAPS["similarity"].copy()
ams_nl = DEFAULT_REG_PARAM_MAPS["nl"].copy()

ams_rig["Metric"] = ["AdvancedMeanSquares"]
ams_aff["Metric"] = ["AdvancedMeanSquares"]
ams_sim["Metric"] = ["AdvancedMeanSquares"]
ams_nl["Metric"] = ["AdvancedMeanSquares"]

DEFAULT_REG_PARAM_MAPS["rigid_ams"] = ams_rig
DEFAULT_REG_PARAM_MAPS["affine_ams"] = ams_aff
DEFAULT_REG_PARAM_MAPS["similarity_ams"] = ams_sim
DEFAULT_REG_PARAM_MAPS["nl_ams"] = ams_nl

# normalized correlation
anc_rig = DEFAULT_REG_PARAM_MAPS["rigid"].copy()
anc_aff = DEFAULT_REG_PARAM_MAPS["affine"].copy()
anc_sim = DEFAULT_REG_PARAM_MAPS["similarity"].copy()
anc_nl = DEFAULT_REG_PARAM_MAPS["nl"].copy()

anc_rig["Metric"] = ["AdvancedNormalizedCorrelation"]
anc_aff["Metric"] = ["AdvancedNormalizedCorrelation"]
anc_sim["Metric"] = ["AdvancedNormalizedCorrelation"]
anc_nl["Metric"] = ["AdvancedNormalizedCorrelation"]

DEFAULT_REG_PARAM_MAPS["rigid_anc"] = anc_rig
DEFAULT_REG_PARAM_MAPS["affine_anc"] = anc_aff
DEFAULT_REG_PARAM_MAPS["similarity_anc"] = anc_sim
DEFAULT_REG_PARAM_MAPS["nl_anc"] = anc_nl
