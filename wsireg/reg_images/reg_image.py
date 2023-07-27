import json
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import itk
import numpy as np
import SimpleITK as sitk

from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.reg_shapes import RegShapes
from wsireg.utils.im_utils import (
    compute_mask_to_bbox,
    contrast_enhance,
    sitk_inv_int,
    sitk_max_int_proj,
    transform_plane,
)
from wsireg.utils.tform_utils import (
    gen_aff_tform_flip,
    gen_rig_to_original,
    gen_rigid_tform_rot,
    gen_rigid_translation,
    prepare_wsireg_transform_data,
)


class RegImage(ABC):
    """Base class for registration images"""

    _path: Union[str, Path]

    # image data
    _dask_image: da.Array
    _reg_image: Union[sitk.Image, itk.Image]
    _mask: Optional[Union[sitk.Image, itk.Image]] = None

    # image dimension information
    _shape: Tuple[int, int, int]
    _n_ch: int
    _im_dtype: np.dtype

    # channel information
    _channel_axis: int
    _is_rgb: bool
    _is_interleaved: bool
    _channel_names: List[str]
    _channel_colors: List[str]

    # scaling information
    _image_res: Union[Tuple[int, int], Tuple[float, float]]

    # reg image preprocessing
    _preprocessing: Optional[ImagePreproParams] = None

    def __init__(
        self, preprocessing: Optional[Union[ImagePreproParams, Dict]] = None
    ):
        if preprocessing:
            if isinstance(preprocessing, ImagePreproParams):
                self._preprocessing = preprocessing
            elif isinstance(preprocessing, dict):
                self._preprocessing = ImagePreproParams(**preprocessing)
        else:
            self._preprocessing = ImagePreproParams()

    @property
    def path(self) -> Union[str, Path]:
        """Path to image file."""
        return self._path

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of image file (C,Y,X) or (Y,X,C) if RGB"""
        return self._shape

    @property
    def n_ch(self) -> int:
        """Number of channels in image."""
        return self._n_ch

    @property
    def im_dtype(self) -> np.dtype:
        """Data type of image"""
        return self._im_dtype

    @property
    def is_rgb(self) -> bool:
        """Whether image is RGB or not."""
        return self._is_rgb

    @property
    def is_interleaved(self) -> bool:
        """Whether RGB image is interleaved or not."""
        return self._is_interleaved

    @property
    def channel_axis(self) -> int:
        """Axis of the channel dimension."""
        return self._channel_axis

    @property
    def image_res(self) -> Union[float, int]:
        """Spacing of image pixels (only isotropic right now)"""
        return self._image_res

    @property
    def channel_names(self) -> List[str]:
        """Name of the channels of the image."""
        return self._channel_names

    @property
    def channel_colors(self) -> List[str]:
        """Colors of the channels."""
        return self._channel_colors

    @property
    def dask_image(self) -> da.Array:
        """Dask representation of the image."""
        return self._dask_image

    @property
    def mask(self) -> Optional[Union[sitk.Image, itk.Image]]:
        """Mask of the image."""
        return self._mask

    @property
    def reg_image(self) -> Union[sitk.Image, itk.Image]:
        """Preprocessed version of image for registration"""
        return self._reg_image

    @property
    def preprocessing(self) -> Optional[ImagePreproParams]:
        """Preprocessing params to make `reg_image`"""
        return self._preprocessing

    def read_mask(
        self, mask: Union[str, Path, sitk.Image, np.ndarray]
    ) -> sitk.Image:
        """
        Read a mask from geoJSON or a binary image.

        Parameters
        ----------
        mask: path to image/geoJSON or image
            Data to be used to make the mask, can be a path to a geoJSON
            or an image file, or a if an np.ndarray, used directly.

        Returns
        -------
        mask: sitk.Image
            Mask image with spacing/size of `reg_image`
        """
        if isinstance(mask, np.ndarray):
            mask = sitk.GetImageFromArray(mask)
        elif isinstance(mask, (str, Path)):
            if Path(mask).suffix.lower() == ".geojson":
                out_shape = self.shape[:2] if self.is_rgb else self.shape[1:]
                mask_shapes = RegShapes(mask)
                mask = mask_shapes.draw_mask(out_shape[::-1], labels=False)
                mask = sitk.GetImageFromArray(mask)
            else:
                mask = sitk.ReadImage(mask)
        elif isinstance(mask, sitk.Image):
            mask = mask

        mask.SetSpacing((self.image_res, self.image_res))

        return mask

    def preprocess_reg_image_intensity(
        self, image: sitk.Image, preprocessing: ImagePreproParams
    ) -> sitk.Image:
        """
        Preprocess image intensity data to single channel image.

        Parameters
        ----------
        image: sitk.Image
            reg_image to be preprocessed
        preprocessing: ImagePreproParams
            Parameters of the preprocessing

        Returns
        -------
        image: sitk.Image
            Preprocessed single-channel image
        """

        if preprocessing.image_type.value == "FL":
            preprocessing.invert_intensity = False
        elif preprocessing.image_type.value == "BF":
            preprocessing.max_int_proj = False
            preprocessing.contrast_enhance = False
            if self.is_rgb:
                preprocessing.invert_intensity = True

        if preprocessing.max_int_proj:
            image = sitk_max_int_proj(image)

        if preprocessing.contrast_enhance:
            image = contrast_enhance(image)

        if preprocessing.invert_intensity:
            image = sitk_inv_int(image)

        if preprocessing.custom_processing:
            for k, v in preprocessing.custom_processing.items():
                print(f"performing preprocessing: {k}")
                image = v(image)

        image.SetSpacing((self.image_res, self.image_res))

        return image

    def preprocess_reg_image_spatial(
        self,
        image: sitk.Image,
        preprocessing: ImagePreproParams,
        imported_transforms=None,
    ) -> Tuple[sitk.Image, List[Dict]]:
        """
        Spatial preprocessing of the reg_image.

        Parameters
        ----------
        image: sitk.Image
            reg_image to be preprocessed
        preprocessing: ImagePreproParams
            Spatial preprocessing parameters
        imported_transforms:
            Not implemented yet..

        Returns
        -------
        image: sitk.Image
            Spatially preprcessed image ready for registration
        transforms: list of transforms
            List of pre-initial transformations
        """

        transforms = []
        original_size = image.GetSize()

        if preprocessing.downsampling > 1:
            print(
                "performing downsampling by factor: {}".format(
                    preprocessing.downsampling
                )
            )
            image.SetSpacing((self.image_res, self.image_res))
            image = sitk.Shrink(
                image,
                (preprocessing.downsampling, preprocessing.downsampling),
            )

            if self._mask is not None:
                self._mask.SetSpacing((self.image_res, self.image_res))
                self._mask = sitk.Shrink(
                    self._mask,
                    (
                        preprocessing.downsampling,
                        preprocessing.downsampling,
                    ),
                )

            image_res = image.GetSpacing()[0]
        else:
            image_res = self.image_res

        if float(preprocessing.rot_cc) != 0.0:
            print(f"rotating counter-clockwise {preprocessing.rot_cc}")
            rot_tform = gen_rigid_tform_rot(
                image, image_res, preprocessing.rot_cc
            )
            (
                composite_transform,
                _,
                final_tform,
            ) = prepare_wsireg_transform_data({"initial": [rot_tform]})

            image = transform_plane(image, final_tform, composite_transform)

            if self._mask is not None:
                self._mask.SetSpacing((image_res, image_res))
                self._mask = transform_plane(
                    self._mask, final_tform, composite_transform
                )
            transforms.append(rot_tform)

        if preprocessing.flip:
            print(f"flipping image {preprocessing.flip.value}")

            flip_tform = gen_aff_tform_flip(
                image, image_res, preprocessing.flip.value
            )

            (
                composite_transform,
                _,
                final_tform,
            ) = prepare_wsireg_transform_data({"initial": [flip_tform]})

            image = transform_plane(image, final_tform, composite_transform)

            if self._mask is not None:
                self._mask.SetSpacing((image_res, image_res))
                self._mask = transform_plane(
                    self._mask, final_tform, composite_transform
                )

            transforms.append(flip_tform)

        if self._mask and preprocessing.crop_to_mask_bbox:
            print("computing mask bounding box")
            if preprocessing.mask_bbox is None:
                mask_bbox = compute_mask_to_bbox(self._mask)
                preprocessing.mask_bbox = mask_bbox

        if preprocessing.mask_bbox:
            print("cropping to mask")
            translation_transform = gen_rigid_translation(
                image,
                image_res,
                preprocessing.mask_bbox.X,
                preprocessing.mask_bbox.Y,
                preprocessing.mask_bbox.WIDTH,
                preprocessing.mask_bbox.HEIGHT,
            )

            (
                composite_transform,
                _,
                final_tform,
            ) = prepare_wsireg_transform_data(
                {"initial": [translation_transform]}
            )

            image = transform_plane(image, final_tform, composite_transform)

            self.original_size_transform = gen_rig_to_original(
                original_size, deepcopy(translation_transform)
            )

            if self._mask is not None:
                self._mask.SetSpacing((image_res, image_res))
                self._mask = transform_plane(
                    self._mask, final_tform, composite_transform
                )
            transforms.append(translation_transform)

        return image, transforms

    def preprocess_image(self, reg_image: sitk.Image) -> None:
        """
        Run full intensity and spatial preprocessing. Creates the `reg_image` attribute

        Parameters
        ----------
        reg_image: sitk.Image
            Raw form of image to be preprocessed

        """

        reg_image = self.preprocess_reg_image_intensity(
            reg_image, self.preprocessing
        )

        if reg_image.GetDepth() >= 1:
            raise ValueError(
                "preprocessing did not result in a single image plane\n"
                "multi-channel or 3D image return"
            )

        if reg_image.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError(
                "preprocessing did not result in a single image plane\n"
                "multi-component / RGB(A) image returned"
            )

        reg_image, pre_reg_transforms = self.preprocess_reg_image_spatial(
            reg_image, self.preprocessing, self.pre_reg_transforms
        )

        if len(pre_reg_transforms) > 0:
            self.pre_reg_transforms = pre_reg_transforms

        self._reg_image = reg_image

    def reg_image_sitk_to_itk(self, cast_to_float32: bool = True) -> None:
        """
        Convert SimpleITK to ITK for use in ITKElastix.

        Parameters
        ----------
        cast_to_float32: bool
            Whether to make image float32 for ITK, needs to be true for registration.

        """
        origin = self._reg_image.GetOrigin()
        spacing = self._reg_image.GetSpacing()
        # direction = image.GetDirection()
        is_vector = self._reg_image.GetNumberOfComponentsPerPixel() > 1
        if cast_to_float32 is True:
            self._reg_image = sitk.Cast(self._reg_image, sitk.sitkFloat32)
            self._reg_image = sitk.GetArrayFromImage(self._reg_image)
        else:
            self._reg_image = sitk.GetArrayFromImage(self._reg_image)

        self._reg_image = itk.GetImageFromArray(
            self._reg_image, is_vector=is_vector
        )
        self._reg_image.SetOrigin(origin)
        self._reg_image.SetSpacing(spacing)

        if self._mask is not None:
            origin = self._mask.GetOrigin()
            spacing = self._mask.GetSpacing()
            # direction = image.GetDirection()
            is_vector = self._mask.GetNumberOfComponentsPerPixel() > 1
            if cast_to_float32 is True:
                self._mask = sitk.Cast(self._mask, sitk.sitkFloat32)
                self._mask = sitk.GetArrayFromImage(self._mask)
            else:
                self._mask = sitk.GetArrayFromImage(self._mask)

            self._mask = itk.GetImageFromArray(self._mask, is_vector=is_vector)
            self._mask.SetOrigin(origin)
            self._mask.SetSpacing(spacing)

            mask_im_type = itk.Image[itk.UC, 2]
            self._mask = itk.binary_threshold_image_filter(
                self._mask,
                lower_threshold=1,
                inside_value=1,
                ttype=(type(self._mask), mask_im_type),
            )

    @staticmethod
    def _get_all_cache_data_fps(output_dir: Union[str, Path], image_tag: str):
        """Get cached directories"""
        output_dir = Path(output_dir)

        out_image_fp = output_dir / f"{image_tag}_prepro.tiff"
        out_params_fp = output_dir / f"{image_tag}_preprocessing_params.json"
        out_mask_fp = output_dir / f"{image_tag}_prepro_mask.tiff"
        out_init_tform_fp = output_dir / f"{image_tag}_init_tforms.json"
        out_osize_tform_fp = output_dir / f"{image_tag}_orig_size_tform.json"

        return (
            out_image_fp,
            out_params_fp,
            out_mask_fp,
            out_init_tform_fp,
            out_osize_tform_fp,
        )

    def check_cache_preprocessing(
        self, output_dir: Union[str, Path], image_tag: str
    ):
        """

        Parameters
        ----------
        output_dir: path
            Where cached data is on disk
        image_tag:
            Tag of the image modality

        Returns
        -------
        prepro_flag: bool
            Whether a preprocessed version of the image exists in the cache.
        """
        (
            out_image_fp,
            out_params_fp,
            out_mask_fp,
            _,
            _,
        ) = self._get_all_cache_data_fps(output_dir, image_tag)

        if out_image_fp.exists() and out_params_fp.exists():
            cached_preprocessing = ImagePreproParams.parse_file(out_params_fp)
            return self.preprocessing == cached_preprocessing
        else:
            return False

    def cache_image_data(
        self, output_dir: Union[str, Path], image_tag: str, check: bool = True
    ) -> None:
        """
        Save preprocessed image data to a cache in WsiReg2D.
        Parameters
        ----------
        output_dir: path
            Where cached data is on disk
        image_tag:
            Tag of the image modality
        check: bool
            Whether to check for existence of data

        """

        (
            out_image_fp,
            out_params_fp,
            out_mask_fp,
            out_init_tform_fp,
            out_osize_tform_fp,
        ) = self._get_all_cache_data_fps(output_dir, image_tag)

        if check:
            read_from_cache = self.check_cache_preprocessing(
                output_dir, image_tag
            )
        else:
            read_from_cache = False

        if not read_from_cache:
            print(f"Writing preprocessed image for {image_tag}")
            sitk.WriteImage(
                self.reg_image, str(out_image_fp), useCompression=True
            )
            print(f"Finished writing preprocessed image for {image_tag}")
            json.dump(
                deepcopy(
                    self.preprocessing.dict(
                        exclude_none=True, exclude_defaults=True
                    )
                ),
                open(out_params_fp, "w"),
                cls=NpEncoder,
            )
            json.dump(self.pre_reg_transforms, open(out_init_tform_fp, "w"))

            if self._mask is not None:
                print(f"Writing preprocessed mask for {image_tag}")
                sitk.WriteImage(
                    self.mask, str(out_mask_fp), useCompression=True
                )
                print(f"Finished writing preprocessed mask for {image_tag}")

            if self.original_size_transform:
                json.dump(
                    self.original_size_transform, open(out_osize_tform_fp, "w")
                )

    def load_from_cache(self, output_dir: Union[str, Path], image_tag: str):
        """
        Read in preprocessed data from the cache folder.
        Parameters
        ----------
        output_dir: path
            Where cached data is on disk
        image_tag:
            Tag of the image modality

        Returns
        -------
        from_cache_flag: bool
            Whether data was read from cache
        """
        (
            image_fp,
            params_fp,
            mask_fp,
            init_tform_fp,
            osize_tform_fp,
        ) = self._get_all_cache_data_fps(output_dir, image_tag)

        read_from_cache = self.check_cache_preprocessing(output_dir, image_tag)

        if read_from_cache:
            self._reg_image = sitk.ReadImage(str(image_fp))
            self._preprocessing = ImagePreproParams(
                **json.load(open(params_fp, "r"))
            )
            self.pre_reg_transforms = json.load(open(init_tform_fp, "r"))
            if osize_tform_fp.exists():
                self.original_size_transform = json.load(
                    open(osize_tform_fp, "r")
                )

            if mask_fp.exists():
                self._mask = sitk.ReadImage(str(mask_fp))
            return True
        else:
            return False

    @staticmethod
    def load_orignal_size_transform(
        output_dir: Union[str, Path], image_tag: str
    ):
        """
        Read original size transform from cache.

        Parameters
        ----------
        output_dir: path
            Where cached data is on disk
        image_tag:
            Tag of the image modality

        Returns
        -------
        osize_tform: list
            Original size transform or empty

        """
        (
            _,
            _,
            _,
            init_tform_fp,
            _,
        ) = RegImage._get_all_cache_data_fps(output_dir, image_tag)

        if init_tform_fp.exists():
            return [json.load(open(init_tform_fp, "r"))]
        else:
            return []


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
