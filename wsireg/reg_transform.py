from warnings import warn

import numpy as np
import SimpleITK as sitk

from wsireg.utils.tform_conversion import convert_to_itk


class RegTransform:
    def __init__(self, elastix_transform):
        self.elastix_transform = elastix_transform
        self.itk_transform = convert_to_itk(self.elastix_transform)

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

    def compute_inverse_nonlinear(self):

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
        use_np_ordering=False,
        n_dim=3,
        use_inverse=False,
        to_px_idx=False,
    ):
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
