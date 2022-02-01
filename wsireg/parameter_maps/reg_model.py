from enum import Enum, EnumMeta
from pathlib import Path
from typing import Dict, List, Tuple, Union

from wsireg.parameter_maps.reg_params import DEFAULT_REG_PARAM_MAPS

DEFAULT_REG_PARAM_MAPS.keys()
PATH_LIKE = Union[str, Path]


def _elx_lineparser(
    line: str,
) -> Union[Tuple[str, List[str]], Tuple[None, None]]:
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


def _read_elastix_parameter_file(
    elx_param_fp: PATH_LIKE,
) -> Dict[str, List[str]]:
    with open(
        elx_param_fp,
        "r",
    ) as f:
        lines = f.readlines()
    parameters = {}
    for line in lines:
        k, v = _elx_lineparser(line)
        if k is not None:
            parameters.update({k: v})
    return parameters


class RegModelMeta(EnumMeta):
    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except (TypeError, KeyError):
            if isinstance(name, (str, Path)) and Path(name).exists():
                return _read_elastix_parameter_file(name)
            else:
                raise ValueError(
                    "unrecognized registration parameter, please provide"
                    "file path to elastix transform parameters or specify one of "
                    f"{[i.name for i in self]}"
                )


class RegModel(dict, Enum, metaclass=RegModelMeta):
    rigid: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["rigid"]
    affine: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["affine"]
    similarity: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["similarity"]
    nl: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl"]
    fi_correction: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS[
        "fi_correction"
    ]
    nl_reduced: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl-reduced"]
    nl_mid: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl-mid"]
    nl2: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl2"]
    rigid_expanded: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS[
        "rigid-expanded"
    ]
    rigid_test: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["rigid_test"]
    affine_test: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["affine_test"]
    similarity_test: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS[
        "similarity_test"
    ]
    nl_test: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl_test"]
    rigid_ams: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["rigid_ams"]
    affine_ams: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["affine_ams"]
    similarity_ams: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS[
        "similarity_ams"
    ]
    nl_ams: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl_ams"]
    rigid_anc: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["rigid_anc"]
    affine_anc: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["affine_anc"]
    similarity_anc: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS[
        "similarity_anc"
    ]
    nl_anc: Dict[str, List[str]] = DEFAULT_REG_PARAM_MAPS["nl_anc"]
