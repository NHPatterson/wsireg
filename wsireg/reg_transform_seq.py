import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from wsireg.reg_transform import RegTransform
from wsireg.utils.tform_utils import ELX_TO_ITK_INTERPOLATORS


class RegTransformSeq:
    reg_transforms: List[RegTransform] = []
    resampler: Optional[sitk.ResampleImageFilter] = None
    composed_linear_mats: Optional[Dict[str, np.ndarray]] = None
    reg_transforms_itk_order: List[RegTransform] = []

    def __init__(
        self,
        reg_transforms: Optional[
            Union[str, Path, Dict[str, List[str]]]
        ] = None,
        transform_seq_idx: Optional[List[int]] = None,
    ) -> None:

        self._transform_seq_idx = []

        if reg_transforms:
            self.add_transforms(
                reg_transforms, transform_seq_idx=transform_seq_idx
            )
        else:
            self._composite_transform = None
            self._n_transforms = 0

    def add_transforms(self, transforms, transform_seq_idx=None):
        if isinstance(transforms, (str, Path, dict)):
            tform_list, tform_idx = read_wsireg_transform(transforms)
            self.transform_seq_idx = tform_idx
            reg_transforms = [RegTransform(t) for t in tform_list]
            self.reg_transforms = self.reg_transforms + reg_transforms

        elif isinstance(transforms, (list, RegTransform)):
            if isinstance(transforms, RegTransform):
                transforms = [transforms]
            self.reg_transforms = self.reg_transforms + transforms
            self.transform_seq_idx = transform_seq_idx

        self._update_transform_properties()

    @property
    def composite_transform(self) -> sitk.CompositeTransform:
        return self._composite_transform

    @composite_transform.setter
    def composite_transform(self, transforms):
        self._composite_transform = transforms

    @property
    def transform_seq_idx(self) -> List[int]:
        return self._transform_seq_idx

    @transform_seq_idx.setter
    def transform_seq_idx(self, transform_seq):
        if len(self._transform_seq_idx) > 0:
            reindex_val = np.max(self._transform_seq_idx) + 1
        else:
            reindex_val = 0

        transform_seq = [x + reindex_val for x in transform_seq]
        self._transform_seq_idx = self._transform_seq_idx + transform_seq

    @property
    def n_transforms(self) -> int:
        return self._n_transforms

    @n_transforms.setter
    def n_transforms(self) -> None:
        self._n_transforms = len(self.reg_transforms)

    @property
    def output_size(self) -> Tuple[int, int]:
        return self._output_size

    @output_size.setter
    def output_size(self, new_size: Tuple[int, int]) -> None:
        self._output_size = new_size

    @property
    def output_spacing(self) -> Union[Tuple[float, float], Tuple[int, int]]:
        return self._output_spacing

    @output_size.setter
    def output_size(
        self, new_spacing: Union[Tuple[float, float], Tuple[int, int]]
    ) -> None:
        self._output_spacing = new_spacing

    def set_output_spacing(
        self, spacing: Union[Tuple[float, float], Tuple[int, int]]
    ) -> None:

        output_size_scaling = np.asarray(self._output_spacing) / np.asarray(
            spacing
        )
        new_output_size = np.ceil(
            np.multiply(self._output_size, output_size_scaling)
        )
        new_output_size = tuple([int(i) for i in new_output_size])

        self._output_spacing = spacing
        self._output_size = new_output_size

        self._build_resampler()

    def _update_transform_properties(self) -> None:
        self._output_size = self.reg_transforms[-1].output_size
        self._output_spacing = self.reg_transforms[-1].output_spacing
        self._build_transform_data()

    def _build_transform_data(self) -> None:
        self._build_composite_transform(
            self.reg_transforms, self.transform_seq_idx
        )
        self._build_resampler()

    def _build_composite_transform(
        self, reg_transforms, reg_transform_seq_idx
    ) -> None:

        composite_index = []
        for unique_idx in np.unique(reg_transform_seq_idx):
            in_seq_tform_idx = np.where(reg_transform_seq_idx == unique_idx)[0]
            if len(in_seq_tform_idx) > 1:
                composite_index = composite_index + list(
                    in_seq_tform_idx[::-1]
                )
            else:
                composite_index = composite_index + list(in_seq_tform_idx)

        composite_transform = sitk.CompositeTransform(2)

        for tform_idx in composite_index:
            composite_transform.AddTransform(
                reg_transforms[tform_idx].itk_transform
            )

        self._composite_transform = composite_transform
        self.reg_transforms_itk_order = [
            self.reg_transforms[i] for i in composite_index
        ]

    def _build_resampler(self) -> None:
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(self.reg_transforms[-1].output_origin)
        resampler.SetOutputDirection(self.reg_transforms[-1].output_direction)
        resampler.SetSize(self.output_size)
        resampler.SetOutputSpacing(self.output_spacing)

        interpolator = ELX_TO_ITK_INTERPOLATORS.get(
            self.reg_transforms[-1].resample_interpolator
        )
        resampler.SetInterpolator(interpolator)
        resampler.SetTransform(self.composite_transform)
        self.resampler = resampler

    def transform_points(
        self, pt_data: np.ndarray, px_idx=True, source_res=1, output_idx=True
    ) -> np.ndarray:
        tformed_pts = []
        for pt in pt_data:
            if px_idx is True:
                pt = pt * source_res
            for idx, t in enumerate(self.reg_transforms):
                if idx == 0:
                    t_pt = t.inverse_transform.TransformPoint(pt)
                else:
                    t_pt = t.inverse_transform.TransformPoint(t_pt)
            t_pt = np.array(t_pt)

            if output_idx is True:
                t_pt *= 1 / self._output_spacing[0]
            tformed_pts.append(t_pt)

        return np.stack(tformed_pts)

    def append(self, other) -> None:
        self.add_transforms(other.reg_transforms, other.transform_seq_idx)

    def write_transforms(self, output_path: Union[str, Path]):
        return


def read_wsireg_transform(
    parameter_data: Union[str, Path, Dict[Any, Any]]
) -> Tuple[List[Dict[str, List[str]]], List[int]]:

    if isinstance(parameter_data, (str, Path)):
        parameter_data_in = json.load(open(parameter_data, "r"))
    else:
        parameter_data_in = parameter_data

    transform_list = []
    transform_list_seq_id = []

    seq_idx = 0
    for k, v in parameter_data_in.items():
        if k == "initial":
            if isinstance(v, dict):
                transform_list.append(v)
                transform_list_seq_id.append(seq_idx)
                seq_idx += 1
            elif isinstance(v, list):
                for init_tform in v:
                    transform_list.append(init_tform)
                    transform_list_seq_id.append(seq_idx)
                    seq_idx += 1
        else:
            if isinstance(v, dict):
                transform_list.append(v)
                transform_list_seq_id.append(seq_idx)
                seq_idx += 1
            elif isinstance(v, list):
                for tform in v:
                    transform_list.append(tform)
                    transform_list_seq_id.append(seq_idx)
                seq_idx += 1

    return transform_list, transform_list_seq_id
