from typing import Dict, Union
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


def _natural_sort(list_to_sort: list) -> list:
    """
    Sort list account for lack of leading zeroes.
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()  # noqa: E731

    alphanum_key = lambda key: [  # noqa: E731
        convert(c) for c in re.split('([0-9]+)', key)
    ]
    return sorted(list_to_sort, key=alphanum_key)


def read_elastix_iteration_data(
    iteration_txt: Union[Path, str]
) -> Dict[str, np.ndarray]:
    """
    Read and parse elastix iteration info.
    Parameters
    ----------
    iteration_txt: str or Path
        File path to an elastix iteration info file

    Returns
    -------
    iteration_dict: Dict[str, np.ndarray]
        dict containing keys of the data in the information: iteration, metric, time, etc.
    """
    with open(iteration_txt, "r") as f:
        iteration_data = f.readlines()

    iteration_data = [it.split("\t") for it in iteration_data[1:]]

    iteration_dict = {
        "iteration": np.array([int(it[0]) for it in iteration_data]),
        "metric": np.array([float(it[1]) for it in iteration_data]),
        "time[a]": np.array([float(it[2]) for it in iteration_data]),
        "step_size": np.array([float(it[3]) for it in iteration_data]),
        "gradient": np.array([float(it[4]) for it in iteration_data]),
        "iter_time": np.array([float(it[5]) for it in iteration_data]),
    }

    return iteration_dict


def read_elastix_iteration_dir(
    registration_dir: Union[Path, str]
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    """
    Read a directory of iteration info and save organize it in a dict for access and plotting.

    Parameters
    ----------
    registration_dir: str or Path
        output directory of the elastix registration data

    Returns
    -------
    all_iteration_data: Dict[int, Dict[int, Dict[str, np.ndarray]]]
        Data for each registration. Keys are 0, 1, 2 for first transform, second transform, etc.
        sub-keys of each top level key are resolution 0, 1, 2, etc.

    """

    iter_info_fps = sorted(Path(registration_dir).glob("IterationInfo*"))
    iter_info_fps = [
        Path(fp) for fp in _natural_sort([str(fp) for fp in iter_info_fps])
    ]

    all_iteration_data = dict()

    for iter_fp in iter_info_fps:
        model_idx = int(iter_fp.name.split(".")[1])
        if model_idx not in all_iteration_data.keys():
            all_iteration_data.update({model_idx: dict()})
        res_idx = int(iter_fp.name.split(".")[2].strip("R"))
        model_res_data = read_elastix_iteration_data(iter_fp)
        all_iteration_data[model_idx].update({res_idx: model_res_data})

    return all_iteration_data


def read_elastix_intermediate_transform_data(
    transform_txt: Union[Path, str]
) -> Dict[str, str]:
    """
    Read transformation data into a dict for each intermediate transform.
    Parameters
    ----------
    transform_txt: str or Path
        file path to the transform parameters file

    Returns
    -------
    elastix_transform_data: Dict[str,str]
        Transform parameters for each transform in the sequence

    """
    with open(transform_txt, "r") as f:
        transform_parameters = f.readlines()

    elastix_transform_data = dict()
    for param in transform_parameters:
        if param[0] == "/":
            continue
        if param[0] == "\n":
            continue
        param = (
            param.replace("(", "")
            .replace(")", "")
            .replace("\n", "")
            .replace('"', "")
        )
        param_name, param_vals = param.split(" ", 1)
        elastix_transform_data.update({param_name: param_vals})

    return elastix_transform_data


def read_elastix_transform_dir(
    registration_dir: Union[Path, str]
) -> Dict[int, Dict[int, Dict[str, str]]]:
    """
    Read an elastix output directory's transformation data.

    Parameters
    ----------
    registration_dir: str or Path
        file path to the elastix output directory

    Returns
    -------
    all_tform_data: Dict[int, Dict[int, Dict[str, str]]]
        Transform paramteter data for each registration. Keys are 0, 1, 2 for first transform, second transform, etc.
        sub-keys of each top level key are resolution 0, 1, 2, etc.

    """
    tform_info_fps = sorted(
        Path(registration_dir).glob("TransformParameters*")
    )
    tform_info_fps = [
        Path(fp) for fp in _natural_sort([str(fp) for fp in tform_info_fps])
    ]

    all_tform_data = dict()

    for tform_fp in tform_info_fps:
        if len(tform_fp.name.split(".")) > 3:
            model_idx = int(tform_fp.name.split(".")[1])
            if model_idx not in all_tform_data.keys():
                all_tform_data.update({model_idx: dict()})
            res_idx = int(tform_fp.name.split(".")[2].strip("R"))
            model_res_data = read_elastix_intermediate_transform_data(tform_fp)
            all_tform_data[model_idx].update({res_idx: model_res_data})

    return all_tform_data


def create_iteration_plot(
    iteration_dict: Dict[str, np.ndarray], plot_title: str
) -> plt.Figure:
    """
    Generate a multi-panel plot of the elastix iteration info.

    Parameters
    ----------
    iteration_dict: Dict[str, np.ndarray]
        dict containing keys of the data in the information: iteration, metric, time, etc.
    plot_title: str
        Main title of the plot

    Returns
    -------
    fig: plt.Figure
        Matplotlib figure object on the multi-panel plot

    """
    fig, ((plt1, plt2), (plt3, plt4), (plt5, plt6)) = plt.subplots(
        3, 2, sharex=True, figsize=(8, 6)
    )
    fig.suptitle(plot_title)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt_pos = {
        "metric": plt1,
        "time[a]": plt2,
        "step_size": plt3,
        "gradient": plt4,
        "iter_time": plt5,
    }

    for k, v in iteration_dict.items():
        if k == "iteration":
            continue
        x_data = iteration_dict["iteration"]
        y_data = v
        x_name = "iteration"
        y_name = k
        plt_pos[y_name].plot(x_data, y_data)
        plt_pos[y_name].set(xlabel=x_name, ylabel=y_name)
        if k in ["step_size", "gradient"]:
            plt_pos[y_name].set_yscale('log')

    plt6.axis('off')

    return fig


def write_iteration_plots(
    all_iteration_data: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    data_key: str,
    output_dir: Union[str, Path],
) -> None:
    """
    Write all plots for an elastix output folder to a folder.

    Parameters
    ----------
    all_iteration_data: Dict[int, Dict[int, Dict[str, np.ndarray]]]
        Data for each registration. Keys are 0, 1, 2 for first transform, second transform, etc.
        sub-keys of each top level key are resolution 0, 1, 2, etc.
    data_key: str
        bit of text indicating which images are registered
    output_dir: str or Path
        Output path of the registration

    Returns
    -------
    None

    """
    for model_idx, model_data in all_iteration_data.items():
        for res_idx, res_data in model_data.items():
            plot_title = f"Registration info for {data_key}- transform idx {model_idx} - resolution {res_idx}"
            output_filepath = (
                Path(output_dir) / f"IterationPlot.{model_idx}.R{res_idx}.png"
            )
            out_fig = create_iteration_plot(res_data, plot_title)
            out_fig.savefig(str(output_filepath))
            plt.close(out_fig)
