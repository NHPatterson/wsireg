from warnings import warn
from typing import Optional
import numpy as np
import SimpleITK as sitk

from wsireg.utils.tform_conversion import convert_to_itk


class RegTransform:
    """Container for elastix transform that manages inversion and other metadata.
    Converts elastix transformation dict to it's SimpleITK representation

    Attributes
    ----------
    elastix_transform: dict
        elastix transform stored in a python dict
    itk_transform: sitk.Transform
        elastix transform in SimpleITK container
    output_spacing: list of float
        Spacing of the targeted image during registration
    output_size: list of int
        Size of the targeted image during registration
    output_direction: list of float
        Direction of the targeted image during registration (not relevant for 2D applications)
    output_origin: list of float
        Origin of the targeted image during registration
    resampler_interpolator: str
        elastix interpolator setting for resampling the image
    is_linear: bool
        Whether the given transform is linear or non-linear (non-rigid)
    inverse_transform: sitk.Transform or None
        Inverse of the itk transform used for transforming from moving to fixed space
        Only calculated for non-rigid transforms when called by `compute_inverse_nonlinear`
        as the process is quite memory and computationally intensive

    """

    def __init__(self, elastix_transform):
        """

        Parameters
        ----------
        elastix_transform: dict
            elastix transform stored in a python dict
        """
        self.elastix_transform: dict = elastix_transform
        self.itk_transform: sitk.Transform = convert_to_itk(
            self.elastix_transform
        )

        self.output_spacing = [
            float(p) for p in self.elastix_transform.get("Spacing")
        ]
        self.output_size = [int(p) for p in self.elastix_transform.get("Size")]
        self.output_origin = [
            float(p) for p in self.elastix_transform.get("Origin")
        ]
        self.output_direction = [
            float(p) for p in self.elastix_transform.get("Direction")
        ]
        self.resample_interpolator = self.elastix_transform.get(
            "ResampleInterpolator"
        )[0]

        self.is_linear = self.itk_transform.IsLinear()

        if self.is_linear is True:
            self.inverse_transform = self.itk_transform.GetInverse()
            transform_name = self.itk_transform.GetName()
            if transform_name == "Euler2DTransform":
                self.inverse_transform = sitk.Euler2DTransform(
                    self.inverse_transform
                )
            elif transform_name == "AffineTransform":
                self.inverse_transform = sitk.AffineTransform(
                    self.inverse_transform
                )
            elif transform_name == "Similarity2DTransform":
                self.inverse_transform = sitk.Similarity2DTransform(
                    self.inverse_transform
                )
        else:
            self.inverse_transform = None

    def compute_inverse_nonlinear(self) -> None:
        """Compute the inverse of a BSpline transform using ITK"""

        tform_to_dfield = sitk.TransformToDisplacementFieldFilter()
        tform_to_dfield.SetOutputSpacing(self.output_spacing)
        tform_to_dfield.SetOutputOrigin(self.output_origin)
        tform_to_dfield.SetOutputDirection(self.output_direction)
        tform_to_dfield.SetSize(self.output_size)

        displacement_field = tform_to_dfield.Execute(self.itk_transform)
        displacement_field = sitk.InvertDisplacementField(displacement_field)
        displacement_field = sitk.DisplacementFieldTransform(
            displacement_field
        )

        self.inverse_transform = displacement_field

    def as_np_matrix(
        self,
        use_np_ordering: bool = False,
        n_dim: int = 3,
        use_inverse: bool = False,
        to_px_idx: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Creates a affine transform matrix as np.ndarray whether the center of rotation
        is 0,0. Optionally in physical or pixel coordinates.
        Parameters
        ----------
        use_np_ordering: bool
            Use numpy ordering of yx (napari-compatible)
        n_dim: int
            Number of dimensions in the affine matrix, using 3 creates a 3x3 array
        use_inverse: bool
            return the inverse affine transformation
        to_px_idx: bool
            return the transformation matrix specified in pixels or physical (microns)

        Returns
        -------
        full_matrix: np.ndarray
            Affine transformation matrix
        """
        if self.is_linear:
            if use_np_ordering is True:
                order = slice(None, None, -1)
            else:
                order = slice(None, None, 1)

            if use_inverse is True:
                transform = self.inverse_transform
            else:
                transform = self.itk_transform

            # pull transform values
            tmatrix = np.array(transform.GetMatrix()[order]).reshape(2, 2)
            center = np.array(transform.GetCenter()[order])
            translation = np.array(transform.GetTranslation()[order])

            if to_px_idx is True:
                phys_to_index = 1 / np.asarray(self.output_spacing).astype(
                    np.float64
                )
                center *= phys_to_index
                translation *= phys_to_index

            # construct matrix
            full_matrix = np.eye(n_dim)
            full_matrix[0:2, 0:2] = tmatrix
            full_matrix[0:2, n_dim - 1] = (
                -np.dot(tmatrix, center) + translation + center
            )

            return full_matrix
        else:
            warn(
                "Non-linear transformations can not be represented converted"
                "to homogenous matrix"
            )
            return None
