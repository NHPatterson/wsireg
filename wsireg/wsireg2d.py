import json
import tempfile
import time
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import shutil

import numpy as np
import yaml

from wsireg.parameter_maps.preprocessing import ImagePreproParams
from wsireg.parameter_maps.reg_model import RegModel
from wsireg.reg_images import MergeRegImage
from wsireg.reg_images.reg_image import RegImage

from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms import RegTransform
from wsireg.reg_transforms import RegTransformSeq
from wsireg.utils.config_utils import parse_check_reg_config
from wsireg.utils.im_utils import ARRAYLIKE_CLASSES
from wsireg.utils.output_utils import (
    read_elastix_iteration_dir,
    read_elastix_transform_dir,
    write_iteration_plots,
)
from wsireg.utils.reg_utils import (
    _prepare_reg_models,
    register_2d_images_itkelx,
    sitk_pmap_to_dict,
)
from wsireg.utils.shape_utils import invert_nonrigid_transforms
from wsireg.utils.tform_utils import identity_elx_transform
from wsireg.writers.merge_ome_tiff_writer import MergeOmeTiffWriter
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.writers.tiled_ome_tiff_writer import OmeTiffTiledWriter


class WsiReg2D(object):
    """
    Class to define a 2D registration graph and execute the registrations and transformations of the graph

    Parameters
    ----------
    project_name: str
        Project name will prefix all output files and directories
    output_dir: str
        Directory where registration data will be stored
    cache_images: bool
        whether to store images as they are preprocessed for registration (if you need to repeat or modify settings
        this will avoid image io and preprocessing)
    config: str or Path
        path to a 2D wsireg YAML configuration

    Attributes
    ----------
    project_name: str
        Global project name, will be appended to all output files and folders
    output_dir: Path
        Directory where registration data will be stored
    image_cache: Path
        Directory where images are cached after preprocessing
    modalities: dict
        dictionary of modality information (file path, spatial res., preprocessing), defines a graph node
    modalities: list
        list of all modality names

    n_modalities: int
        number of modalities (nodes) in the graphs

    reg_paths: dict
        dictionary of a modalities path from node to node

    reg_graph_edges: dict
        generated dictionary of necessary registrations to move modalities to their target modality

    n_registrations: int
        number of explicit registrations (edges) in the graphs

    transformation_paths: dict
        generated dictionary of necessary source - target transformations to transform modalities to their target modality

    transformations: dict
        per modality dictionary containing transformation parameters for each registration

    attachment_images: dict
        images to be transformed along the path of the defined graph, assoicated to a given modality (masks, other registered images)

    attachment_shapes: dict
        shape data attached to a modality to be transformed along the graph

    registration_iter_data: Dict[str,Dict[int, Dict[int, Dict[str, np.ndarray]]]]
        elastix data for each iteration in the registration sorted by transformation model and resolution

    registration_tform_data: Dict[str, Dict[int, Dict[int, Dict[str, str]]]]
        elastix transform data for each resolution sorted by transformation model and resolution

    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        cache_images: bool = True,
        config: Optional[Union[str, Path]] = None,
    ):
        self.project_name: Optional[str] = None
        self.output_dir: Optional[Union[str, Path]] = None
        self.image_cache: Optional[Path] = None

        self.setup_project_output(project_name, output_dir)

        self.cache_images = cache_images

        self.pairwise = False

        self._modalities = dict()
        self._modality_names = []
        self._reg_paths = dict()
        self._reg_graph_edges = []
        self._transform_paths = dict()

        self._transformations = None
        self._preprocessed_image_sizes: Dict[str, Tuple[int, int]] = dict()
        self._preprocessed_image_spacings: Dict[
            str, Union[Tuple[int, int], Tuple[float, float]]
        ] = dict()
        self._transformed_shapes_spacings: Dict[
            str, Tuple[float, float]
        ] = dict()

        self.n_modalities: Optional[int] = None
        self.n_registrations: Optional[int] = None

        self.attachment_images = dict()

        self._shape_sets = dict()
        self._shape_set_names = []

        self.merge_modalities = dict()
        self.original_size_transforms = dict()

        self.registration_iter_data: Dict[
            str, Dict[int, Dict[int, Dict[str, np.ndarray]]]
        ] = dict()
        self.registration_tform_data: Dict[
            str, Dict[int, Dict[int, Dict[str, str]]]
        ] = dict()

        if config:
            self.add_data_from_config(config)

    def setup_project_output(
        self,
        project_name: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Set up the project directory and image cache.

        Parameters
        ----------
        project_name: str
            Project name will prefix all output files and directories
        output_dir: str
            Directory where registration data will be stored


        """
        if project_name is None:
            self.project_name = 'RegProj'
        else:
            self.project_name = project_name

        if output_dir is None:
            output_dir = "./"

        self.output_dir = Path(output_dir)

        self.image_cache = self.output_dir / ".imcache_{}".format(project_name)

    @property
    def modalities(self) -> Dict[str, Any]:
        """Image modality information stored as a dictionary where keys are images and values
        are all image modality related information."""
        return self._modalities

    @modalities.setter
    def modalities(self, modality):
        self._modalities.update(modality)
        self.n_modalities = len(self._modalities)

    @property
    def shape_sets(self):
        """Shape data stored as a dictionary where keys are shape sets and values
        are all shape related information."""
        return self._shape_sets

    @shape_sets.setter
    def shape_sets(self, shape_set):
        self._shape_sets.update(shape_set)

    @property
    def shape_set_names(self) -> List[str]:
        """Name of all shape sets"""
        return self._shape_set_names

    @shape_set_names.setter
    def shape_set_names(self, shape_set_name):
        self._shape_set_names.append(shape_set_name)

    @property
    def modality_names(self) -> List[str]:
        """List of all the modality names."""
        return self._modality_names

    @modality_names.setter
    def modality_names(self, modality_name):
        self._modality_names.append(modality_name)

    def add_modality(
        self,
        modality_name: str,
        image_fp: Union[Path, str, np.ndarray],
        image_res: Union[int, float] = 1,
        channel_names: Optional[List[str]] = None,
        channel_colors: Optional[List[str]] = None,
        preprocessing: Optional[
            Union[ImagePreproParams, Dict[str, Any]]
        ] = None,
        mask: Optional[Union[str, Path, np.ndarray]] = None,
        prepro_dict: Optional[Dict[str, Any]] = None,
        output_res: Optional[
            Union[int, float, Tuple[int, int], Tuple[float, float]]
        ] = None,
    ) -> None:
        """
        Add an image modality (node) to the registration graph

        Parameters
        ----------
        modality_name : str
            Unique name identifier for the modality
        image_fp : str
            file path to the image to be read
        image_res : float
            spatial resolution of image in units per px (i.e. 0.9 um / px)
        channel_names: List[str]
            names for the channels to go into the OME-TIFF
        channel_colors: List[str]
            channels colors for OME-TIFF (not implemented)
        mask: Union[str, Path, np.ndarray]
            path to binary mask (>0 is in) image for registration and/or cropping or a geoJSON with shapes
            that will be processed to a binary mask
        preprocessing: ImagePreproParams
            preprocessing parameters for the modality for registration. Registration images should be a xy single plane
            so many modalities (multi-channel, RGB) must "create" a single channel.
            Defaults: multi-channel images -> max intensity project image
            RGB -> greyscale then intensity inversion (black background, white foreground)
        output_res : Union[Tuple[int,int], Tuple[float,float]]
            change output spacing/resolution when resampling images, default will be the spacing
            of the final target image
        prepro_dict: dict
            deprecated version kept temporarily
        """
        if modality_name in self._modality_names:
            raise ValueError(
                'modality named \"{}\" is already in modality_names'.format(
                    modality_name
                )
            )
        if prepro_dict:
            preprocessing = prepro_dict
            DeprecationWarning(
                "'prepro_dict' argument is deprepcated and will be removed\
                in a future release, use 'preprocessing' instead"
            )
        if preprocessing:
            if isinstance(preprocessing, dict):
                image_prepro = ImagePreproParams(**preprocessing)
            else:
                image_prepro = preprocessing
        else:
            image_prepro = ImagePreproParams()

        if isinstance(output_res, (int, float)):
            output_res = (output_res, output_res)

        self.modalities = {
            modality_name: {
                "image_filepath": image_fp,
                "image_res": image_res,
                "channel_names": channel_names,
                "channel_colors": channel_colors,
                "preprocessing": image_prepro,
                "mask": mask,
                "output_res": output_res,
            }
        }

        self.modality_names = modality_name

    def add_shape_set(
        self,
        attachment_modality: str,
        shape_set_name: str,
        shape_files: List[Union[str, Path]],
        image_res: Union[int, float],
    ) -> None:
        """
        Add a shape set to the graph

        Parameters
        ----------
        attachment_modality : str
            image modality to which the shapes are attached
        shape_set_name : str
            Unique name identifier for the shape set
        shape_files : list of file paths
            list of shape data in geoJSON format or list of dicts containing following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        image_res : float
            spatial resolution of shape data's associated image in units per px (i.e. 0.9 um / px)
        """
        if shape_set_name in self._shape_set_names:
            raise ValueError(
                'shape named \"{}\" is already in shape_set_names'.format(
                    shape_set_name
                )
            )

        self.shape_sets = {
            shape_set_name: {
                "shape_files": shape_files,
                "image_res": image_res,
                "attachment_modality": attachment_modality,
            }
        }

        self.shape_set_names = shape_set_name

    def add_attachment_images(
        self,
        attachment_modality: str,
        modality_name: str,
        image_fp: Union[str, Path],
        image_res: Union[int, float] = 1,
        channel_names: Optional[List[str]] = None,
        channel_colors: Optional[List[str]] = None,
    ) -> None:
        """
        Images which are unregistered between modalities, but are transformed following the path of one of the graph's
        modalities.

        Parameters
        ----------
        attachment_modality : str
            image modality to which the new image are attached
        modality_name : str
            name of the added attachment image
        image_fp : str
            path to the attachment modality, it will be imported and transformed without preprocessing
        image_res : float
            spatial resolution of attachment image data's in units per px (i.e. 0.9 um / px)
        channel_names: List[str]
            names for the channels to go into the OME-TIFF
        channel_colors: List[str]
            channels colors for OME-TIFF (not implemented)
        """
        if attachment_modality not in self.modality_names:
            raise ValueError(
                'attachment modality named \"{}\" not found in modality_names'.format(
                    attachment_modality
                )
            )
        self.add_modality(
            modality_name,
            image_fp,
            image_res,
            channel_names=channel_names,
            channel_colors=channel_colors,
        )
        self.attachment_images[modality_name] = attachment_modality

    def add_attachment_shapes(
        self,
        attachment_modality: str,
        shape_set_name: str,
        shape_files: List[Union[str, Path]],
    ) -> None:
        """
        Add attached shapes

        Parameters
        ----------
        attachment_modality : str
            image modality to which the shapes are attached
        shape_set_name : str
            Unique name identifier for the shape set
        shape_files : list of file paths
            list of shape data in geoJSON format or list of dicts containing following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        """
        if attachment_modality not in self.modality_names:
            raise ValueError(
                'attachment modality \"{}\" for shapes \"{}\" not found in modality_names {}'.format(
                    attachment_modality, shape_set_name, self.modality_names
                )
            )

        image_res = self.modalities[attachment_modality]["image_res"]
        self.add_shape_set(
            attachment_modality, shape_set_name, shape_files, image_res
        )

    @property
    def reg_paths(self) -> Dict[str, List[str]]:
        """Dictionary of paths between modalities by modality name.
        Keys are modality name and the values are a list of modalities it passes through.
        """
        return self._reg_paths

    @reg_paths.setter
    def reg_paths(self, path_values):
        (
            src_modality,
            tgt_modality,
            thru_modality,
            reg_params,
            override_prepro,
        ) = path_values

        if thru_modality != tgt_modality:
            self._reg_paths.update(
                {src_modality: [thru_modality, tgt_modality]}
            )
        else:
            self._reg_paths.update({src_modality: [tgt_modality]})
        if override_prepro:
            if override_prepro.get("source"):
                source_override = ImagePreproParams(
                    **override_prepro.get("source")
                )
            else:
                source_override = None

            if override_prepro.get("target"):
                target_override = ImagePreproParams(
                    **override_prepro.get("target")
                )
            else:
                target_override = None
        else:
            source_override = None
            target_override = None

        self.reg_graph_edges = {
            'modalities': {'source': src_modality, 'target': thru_modality},
            'params': reg_params,
            "source_override": source_override,
            "target_override": target_override,
        }
        self.transform_paths = self._reg_paths

    def add_reg_path(
        self,
        src_modality_name: str,
        tgt_modality_name: str,
        thru_modality: Optional[str] = None,
        reg_params: Union[str, RegModel, List[str], List[RegModel]] = [
            "rigid"
        ],
        override_prepro: dict = {"source": None, "target": None},
    ):
        """
        Add registration path between modalities as well as a thru modality that describes where to attach edges.

        Parameters
        ----------
        src_modality_name : str
            modality that has been added to graph to be transformed to tgt_modality
        tgt_modality_name : str
            modality that has been added to graph that is being aligned to
        thru_modality: str
            modality that has been added to graph by which another should be run through
        reg_params: list of RegModel/str or str
            Elastix registration parameters, from RegModel or as a string corresponding to one of the parameter
            maps enumerated in wsireg.parameter_maps.reg_params.RegModel
        override_prepro: dict
            set specific preprocessing for a given registration edge for the source or target image that will override
            the set modality preprocessing FOR THIS REGISTRATION ONLY.
        """
        if src_modality_name not in self.modality_names:
            raise ValueError("source modality not found!")
        if tgt_modality_name not in self.modality_names:
            raise ValueError("target modality not found!")

        if thru_modality is None:
            self.reg_paths = (
                src_modality_name,
                tgt_modality_name,
                tgt_modality_name,
                reg_params,
                override_prepro,
            )
        else:
            self.reg_paths = (
                src_modality_name,
                tgt_modality_name,
                thru_modality,
                reg_params,
                override_prepro,
            )

    @property
    def reg_graph_edges(self):
        return self._reg_graph_edges

    @reg_graph_edges.setter
    def reg_graph_edges(self, edge):
        self._reg_graph_edges.append(edge)
        self.n_registrations = len(self._reg_graph_edges)

    @property
    def transform_paths(self):
        return self._transform_paths

    @transform_paths.setter
    def transform_paths(self, reg_paths):
        transform_path_dict = {}

        for k, v in reg_paths.items():
            tform_path = self.find_path(k, v[-1])
            if self.pairwise is True:
                tform_path_modalities = tform_path[:1]
            else:
                tform_path_modalities = tform_path
            transform_edges = []
            for modality in tform_path_modalities:
                for edge in self.reg_graph_edges:
                    edge_modality = edge["modalities"]['source']
                    if modality == edge_modality:
                        transform_edges.append(edge["modalities"])
            transform_path_dict.update({k: transform_edges})

        self._transform_paths = transform_path_dict

    def find_path(
        self,
        start_modality: str,
        end_modality: str,
        path: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Find a path from start_modality to end_modality in the graph
        """
        if path is None:
            path = []
        path = path + [start_modality]
        if start_modality == end_modality:
            return path
        if start_modality not in self.reg_paths:
            return None
        for modality in self.reg_paths[start_modality]:
            if modality not in path:
                extended_path = self.find_path(modality, end_modality, path)
                if extended_path:
                    return extended_path
        return None

    def _find_nonreg_modalities(self) -> None:
        registered_modalities = [
            edge.get("modalities").get("source")
            for edge in self.reg_graph_edges
        ]
        non_reg_modalities = list(
            set(self.modality_names).difference(registered_modalities)
        )

        # remove attachment modalities
        for attachment_modality in self.attachment_images.keys():
            non_reg_modalities.pop(
                non_reg_modalities.index(attachment_modality)
            )

        return non_reg_modalities

    def save_config(
        self,
        output_file_path: Optional[Union[str, Path]] = None,
        registered: bool = False,
    ) -> str:
        ts = time.strftime('%Y%m%d-%H%M%S')
        status = "registered" if registered is True else "setup"

        reg_paths = {}
        for idx, edge in enumerate(self.reg_graph_edges):
            src_modality = edge.get("modalities").get("source")
            if len(self.reg_paths[src_modality]) > 1:
                thru_modality = self.reg_paths[src_modality][0]
            else:
                thru_modality = None
            tgt_modality = self.reg_paths[src_modality][-1]
            reg_paths.update(
                {
                    f"reg_path_{idx}": {
                        "src_modality_name": edge.get("modalities").get(
                            "source"
                        ),
                        "tgt_modality_name": tgt_modality,
                        "thru_modality": thru_modality,
                        "reg_params": edge.get("params"),
                    }
                }
            )

        reg_graph_edges = deepcopy(self.reg_graph_edges)

        [rge.pop("transforms", None) for rge in reg_graph_edges]

        modalities_out = deepcopy(self.modalities)
        for mod, data in modalities_out.items():
            if isinstance(data["image_filepath"], ARRAYLIKE_CLASSES):
                data["image_filepath"] = "ArrayLike"
            if isinstance(data["preprocessing"], ImagePreproParams):
                data["preprocessing"] = deepcopy(data["preprocessing"]).dict(
                    exclude_none=True, exclude_defaults=True
                )

        config = {
            "project_name": self.project_name,
            "output_dir": str(self.output_dir),
            "cache_images": self.cache_images,
            "modalities": modalities_out,
            "reg_paths": reg_paths,
            "reg_graph_edges": reg_graph_edges
            if status == "registered"
            else None,
            "original_size_transforms": self.original_size_transforms
            if status == "registered"
            else None,
            "attachment_shapes": self.shape_sets
            if len(self._shape_sets) > 0
            else None,
            "attachment_images": self.attachment_images
            if len(self.attachment_images) > 0
            else None,
            "merge_modalities": self.merge_modalities,
        }

        if not output_file_path:
            output_file_path = (
                self.output_dir
                / f"{ts}-{self.project_name}-configuration-{status}.yaml"
            )

        with open(str(output_file_path), "w") as f:
            yaml.dump(config, f, sort_keys=False)

        return str(output_file_path)

    def register_images(self, parallel=False):
        """
        Start image registration process for all modalities

        Parameters
        ----------
        parallel : bool
            whether to run each edge in parallel (not implemented yet)
        """
        if self.cache_images is True:
            self.image_cache.mkdir(parents=False, exist_ok=True)

        self.save_config(registered=False)

        for reg_edge in self.reg_graph_edges:
            if (
                reg_edge.get("registered") is None
                or reg_edge.get("registered") is False
            ):
                src_name = reg_edge["modalities"]["source"]
                tgt_name = reg_edge["modalities"]["target"]

                src_mod_data = self.modalities[src_name].copy()
                tgt_mod_data = self.modalities[tgt_name].copy()

                src_reg_image = reg_image_loader(
                    src_mod_data["image_filepath"],
                    src_mod_data["image_res"],
                    preprocessing=src_mod_data["preprocessing"],
                    mask=src_mod_data["mask"],
                )

                tgt_reg_image = reg_image_loader(
                    tgt_mod_data["image_filepath"],
                    tgt_mod_data["image_res"],
                    preprocessing=tgt_mod_data["preprocessing"],
                    mask=tgt_mod_data["mask"],
                )

                src_override_prepro = reg_edge.get("source_override")
                tgt_override_prepro = reg_edge.get("target_override")

                src_cached = src_reg_image.check_cache_preprocessing(
                    self.image_cache, src_name
                )
                tgt_cached = src_reg_image.check_cache_preprocessing(
                    self.image_cache, tgt_name
                )

                if not src_cached and not src_override_prepro:
                    src_reg_image.read_reg_image()
                    if self.cache_images:
                        src_reg_image.cache_image_data(
                            self.image_cache, src_name, check=False
                        )
                elif src_override_prepro:
                    if src_override_prepro:
                        src_reg_image._preprocessing = src_override_prepro
                    src_reg_image.read_reg_image()
                    if self.cache_images:
                        src_reg_image.cache_image_data(
                            self.image_cache,
                            f"{src_name}-{tgt_name}-override",
                            check=False,
                        )
                else:
                    src_reg_image.load_from_cache(self.image_cache, src_name)

                if not tgt_cached and not tgt_override_prepro:
                    tgt_reg_image.read_reg_image()
                    if self.cache_images:
                        tgt_reg_image.cache_image_data(
                            self.image_cache, tgt_name, check=False
                        )
                elif tgt_override_prepro:
                    if tgt_override_prepro:
                        tgt_reg_image._preprocessing = tgt_override_prepro
                    tgt_reg_image.read_reg_image()
                    if self.cache_images:
                        src_reg_image.cache_image_data(
                            self.image_cache,
                            f"{tgt_name}-{src_name}-override",
                            check=False,
                        )
                else:
                    tgt_reg_image.load_from_cache(self.image_cache, tgt_name)

                self._preprocessed_image_spacings.update(
                    {src_name: src_reg_image.reg_image.GetSpacing()}
                )
                self._preprocessed_image_spacings.update(
                    {tgt_name: tgt_reg_image.reg_image.GetSpacing()}
                )
                self._preprocessed_image_sizes.update(
                    {src_name: src_reg_image.reg_image.GetSize()}
                )
                self._preprocessed_image_sizes.update(
                    {tgt_name: tgt_reg_image.reg_image.GetSize()}
                )

                reg_params = reg_edge["params"]

                output_path = (
                    self.output_dir
                    / "{}-{}_to_{}_reg_output".format(
                        self.project_name,
                        reg_edge["modalities"]["source"],
                        reg_edge["modalities"]["target"],
                    )
                )

                output_path.mkdir(parents=False, exist_ok=True)

                reg_params_prepared = _prepare_reg_models(reg_params)

                reg_tforms = register_2d_images_itkelx(
                    src_reg_image,
                    tgt_reg_image,
                    reg_params_prepared,
                    output_path,
                )

                reg_tforms = [sitk_pmap_to_dict(tf) for tf in reg_tforms]

                initial_transforms = src_reg_image.pre_reg_transforms

                if initial_transforms:
                    initial_transforms_rt = [
                        RegTransform(t) for t in initial_transforms
                    ]
                    initial_transforms_idx = [
                        idx for idx, _ in enumerate(initial_transforms_rt)
                    ]
                    initial_rt_seq = RegTransformSeq(
                        initial_transforms_rt, initial_transforms_idx
                    )

                reg_tforms_rt = [RegTransform(t) for t in reg_tforms]
                reg_tforms_idx = [0 for _ in reg_tforms_rt]
                reg_rt_seq = RegTransformSeq(reg_tforms_rt, reg_tforms_idx)

                reg_edge["transforms"] = {
                    'initial': initial_rt_seq if initial_transforms else None,
                    'registration': reg_rt_seq,
                }

                self.original_size_transforms.update(
                    {tgt_name: tgt_reg_image.original_size_transform}
                )

                reg_edge["registered"] = True
                data_key = f'{reg_edge["modalities"]["source"]}_to_{reg_edge["modalities"]["target"]}'

                all_transform_data = read_elastix_transform_dir(output_path)
                all_iteration_data = read_elastix_iteration_dir(output_path)
                write_iteration_plots(
                    all_iteration_data, data_key, output_path
                )

                self.registration_iter_data.update(
                    {data_key: all_iteration_data}
                )
                self.registration_tform_data.update(
                    {data_key: all_transform_data}
                )

        self.transformations = self.reg_graph_edges

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, reg_graph_edges):
        self._transformations = self._collate_transformations()

    def add_merge_modalities(self, merge_name, modalities):
        for modality in modalities:
            try:
                self.modalities[modality]
            except KeyError:
                raise ValueError(
                    f"Modality for merger [{modality}] is not a modality "
                    f"within the graph, current modalitles : "
                    f"{self.modality_names}"
                )
            self.merge_modalities.update({merge_name: modalities})

    def _generate_reg_transforms(self):
        self._reg_graph_edges["reg_transforms"]

    def _collate_transformations(self):
        transforms = dict()
        edge_modality_pairs = [v['modalities'] for v in self.reg_graph_edges]
        for modality, tform_edges in self.transform_paths.items():
            full_tform_seq = RegTransformSeq()
            for idx, tform_edge in enumerate(tform_edges):
                reg_edge_tforms = self.reg_graph_edges[
                    edge_modality_pairs.index(tform_edge)
                ]["transforms"]
                if idx == 0:
                    transforms[modality] = {
                        f"initial-{tform_edges[idx]['source']}": reg_edge_tforms[
                            'initial'
                        ],
                        f"{str(idx).zfill(3)}-to-{tform_edges[idx]['target']}": reg_edge_tforms[
                            'registration'
                        ],
                    }
                    if reg_edge_tforms['initial']:
                        full_tform_seq.append(reg_edge_tforms['initial'])
                    full_tform_seq.append(reg_edge_tforms['registration'])
                else:
                    transforms[modality][
                        f"{str(idx).zfill(3)}-to-{tform_edges[idx]['target']}"
                    ] = reg_edge_tforms['registration']
                    full_tform_seq.append(reg_edge_tforms['registration'])

                transforms[modality]["full-transform-seq"] = full_tform_seq

        return transforms

    def _prepare_nonreg_image_transform(
        self,
        modality_key,
        attachment=False,
        attachment_modality=None,
        to_original_size=True,
    ):
        print(
            "preparing transforms for non-registered modality : {} ".format(
                modality_key
            )
        )
        output_path = self.output_dir / "{}-{}_registered".format(
            self.project_name, modality_key
        )

        if attachment:
            im_data_key = copy(modality_key)
            modality_key = attachment_modality

        transformations = None

        im_data = self.modalities[modality_key]

        if (
            im_data["preprocessing"].rot_cc != 0
            or im_data["preprocessing"].flip
            or im_data["preprocessing"].crop_to_mask_bbox
            or im_data["preprocessing"].mask_bbox
        ):
            im_initial_transforms = RegImage.load_orignal_size_transform(
                self.image_cache, modality_key
            )

            if any(im_initial_transforms):
                transformations = RegTransformSeq(
                    [RegTransform(t) for t in im_initial_transforms[0]],
                    transform_seq_idx=[
                        idx for idx in range(len(im_initial_transforms))
                    ],
                )

        if to_original_size and self.original_size_transforms[modality_key]:
            o_size_tform = self.original_size_transforms[modality_key]
            if isinstance(o_size_tform, list):
                o_size_tform = o_size_tform[0]

            orig_size_rt = RegTransformSeq(
                RegTransform(o_size_tform),
                transform_seq_idx=[0],
            )
            if transformations:
                transformations.append(orig_size_rt)
            else:
                transformations = orig_size_rt

        if im_data["preprocessing"].downsampling > 1 and transformations:
            if not im_data["output_res"]:
                output_spacing_target = im_data["image_res"]
                transformations.set_output_spacing(
                    (output_spacing_target, output_spacing_target)
                )
            else:
                transformations.set_output_spacing(im_data["output_res"])

        elif im_data["preprocessing"].downsampling > 1 and not transformations:
            transformations = RegTransformSeq(
                [
                    RegTransform(
                        identity_elx_transform(
                            self._preprocessed_image_sizes[modality_key],
                            self._preprocessed_image_spacings[modality_key],
                        )
                    )
                ],
                transform_seq_idx=[0],
            )

            if not im_data["output_res"]:
                output_spacing_target = im_data["image_res"]
                transformations.set_output_spacing(
                    (output_spacing_target, output_spacing_target)
                )
            else:
                transformations.set_output_spacing(im_data["output_res"])

        if attachment:
            im_data = self.modalities[im_data_key]

        return im_data, transformations, output_path

    def _prepare_reg_image_transform(
        self,
        edge_key,
        attachment=False,
        attachment_modality=None,
        to_original_size=True,
    ):
        if attachment:
            final_modality = self.reg_paths[attachment_modality][-1]
            transformations = copy(
                self.transformations[attachment_modality]["full-transform-seq"]
            )
        else:
            final_modality = self.reg_paths[edge_key][-1]
            transformations = copy(
                self.transformations[edge_key]["full-transform-seq"]
            )

        output_path = self.output_dir / "{}-{}_to_{}_registered".format(
            self.project_name,
            edge_key,
            final_modality,
        )

        if attachment:
            im_data_key = copy(edge_key)
            edge_key = attachment_modality

        im_data = self.modalities[edge_key]

        if (
            self.original_size_transforms.get(final_modality)
            and to_original_size
        ):
            print("adding transform to original size")
            original_size_transform = self.original_size_transforms[
                final_modality
            ]
            if isinstance(original_size_transform, list):
                original_size_transform = original_size_transform[0]

            orig_size_rt = RegTransformSeq(
                RegTransform(original_size_transform), transform_seq_idx=[0]
            )
            transformations.append(orig_size_rt)

        if im_data["preprocessing"].downsampling > 1:
            if not im_data["output_res"]:
                output_spacing_target = self.modalities[final_modality][
                    "image_res"
                ]
                transformations.set_output_spacing(
                    (output_spacing_target, output_spacing_target)
                )
            else:
                transformations.set_output_spacing(im_data["output_res"])

        elif im_data["output_res"]:
            transformations.set_output_spacing(im_data["output_res"])

        if attachment:
            im_data = self.modalities[im_data_key]

        return im_data, transformations, output_path

    def _transform_write_image(
        self, im_data, transformations, output_path, file_writer="ome.tiff"
    ):
        tfregimage = reg_image_loader(
            im_data["image_filepath"],
            im_data["image_res"],
            channel_names=im_data.get("channel_names"),
            channel_colors=im_data.get("channel_colors"),
        )

        ometiffwriter = OmeTiffWriter(
            tfregimage, reg_transform_seq=transformations
        )

        if (
            file_writer == "ome.tiff-bytile"
            and ometiffwriter.reg_image.reader not in ["czi", "sitk"]
        ):
            ometiffwriter = OmeTiffTiledWriter(
                tfregimage, reg_transform_seq=transformations
            )
            im_fp = ometiffwriter.write_image_by_tile(
                output_path.stem,
                output_dir=str(self.output_dir),
            )
        else:
            im_fp = ometiffwriter.write_image_by_plane(
                output_path.stem,
                output_dir=str(self.output_dir),
            )

        return im_fp

    def _transform_write_merge_images(self, to_original_size=True):
        def determine_attachment(sub_image):
            if sub_image in self.attachment_images.keys():
                attachment_modality = self.attachment_images[sub_image]
                attachment = True
            else:
                attachment_modality = None
                attachment = False
            return attachment, attachment_modality

        for merge_name, sub_images in self.merge_modalities.items():
            im_fps = []
            im_res = []
            im_ch_names = []
            transformations = []

            non_reg_modalities = self._find_nonreg_modalities()

            for sub_image in sub_images:
                attachment, attachment_modality = determine_attachment(
                    sub_image
                )
                im_data = self.modalities[sub_image]
                im_fps.append(im_data["image_filepath"])
                im_res.append(im_data["image_res"])
                im_ch_names.append(im_data.get("channel_names"))
                if (
                    sub_image not in non_reg_modalities
                    and attachment_modality not in non_reg_modalities
                ):
                    (
                        _,
                        sub_im_transforms,
                        _,
                    ) = self._prepare_reg_image_transform(
                        sub_image,
                        attachment=attachment,
                        attachment_modality=attachment_modality,
                        to_original_size=to_original_size,
                    )

                else:
                    (
                        _,
                        sub_im_transforms,
                        _,
                    ) = self._prepare_nonreg_image_transform(
                        sub_image,
                        attachment=attachment,
                        attachment_modality=attachment_modality,
                        to_original_size=to_original_size,
                    )

                transformations.append(sub_im_transforms)

        output_path = self.output_dir / "{}-{}_merged-registered".format(
            self.project_name,
            merge_name,
        )

        merge_regimage = MergeRegImage(
            im_fps,
            im_res,
            channel_names=im_ch_names,
        )

        merge_ometiffwriter = MergeOmeTiffWriter(
            merge_regimage, reg_transform_seqs=transformations
        )

        im_fp = merge_ometiffwriter.merge_write_image_by_plane(
            output_path.stem,
            sub_images,
            output_dir=str(self.output_dir),
        )
        return im_fp

    def transform_images(
        self,
        file_writer="ome.tiff",
        transform_non_reg=True,
        remove_merged=True,
        to_original_size=True,
    ):
        """
        Transform and write images to disk after registration. Also transforms all attachment images

        Parameters
        ----------
        file_writer : str
            output type to use, sitk writes a single resolution tiff, "zarr" writes an ome-zarr multiscale
            zarr store
        transform_non_reg : bool
            whether to write the images that aren't transformed during registration as well
        remove_merged: bool
            whether to remove images that are stored in merged store, if True, images that are merged
            will not be written as individual images as well
        to_original_size: bool
            write images that have been cropped for registration back to their original coordinate space
        """
        image_fps = []

        if all(
            [reg_edge.get("registered") for reg_edge in self.reg_graph_edges]
        ):
            # prepare workflow
            merge_modalities = []
            if len(self.merge_modalities.keys()) > 0:
                for k, v in self.merge_modalities.items():
                    merge_modalities.extend(v)

            reg_path_keys = list(self.reg_paths.keys())
            nonreg_keys = self._find_nonreg_modalities()

            if remove_merged:
                for merge_mod in merge_modalities:
                    try:
                        m_idx = reg_path_keys.index(merge_mod)
                        reg_path_keys.pop(m_idx)
                    except ValueError:
                        pass
                    try:
                        m_idx = nonreg_keys.index(merge_mod)
                        nonreg_keys.pop(m_idx)
                    except ValueError:
                        pass

            for modality in reg_path_keys:
                (
                    im_data,
                    transformations,
                    output_path,
                ) = self._prepare_reg_image_transform(
                    modality,
                    attachment=False,
                    to_original_size=to_original_size,
                )

                im_fp = self._transform_write_image(
                    im_data,
                    transformations,
                    output_path,
                    file_writer=file_writer,
                )
                image_fps.append(im_fp)

            for (
                modality,
                attachment_modality,
            ) in self.attachment_images.items():
                if modality in merge_modalities and remove_merged:
                    continue
                if attachment_modality in self._find_nonreg_modalities():
                    (
                        im_data,
                        transformations,
                        output_path,
                    ) = self._prepare_nonreg_image_transform(
                        modality,
                        attachment=True,
                        attachment_modality=attachment_modality,
                        to_original_size=to_original_size,
                    )
                else:
                    (
                        im_data,
                        transformations,
                        output_path,
                    ) = self._prepare_reg_image_transform(
                        modality,
                        attachment=True,
                        attachment_modality=attachment_modality,
                        to_original_size=to_original_size,
                    )

                im_fp = self._transform_write_image(
                    im_data,
                    transformations,
                    output_path,
                    file_writer=file_writer,
                )
                image_fps.append(im_fp)

            if len(self.merge_modalities.items()) > 0:
                im_fp = self._transform_write_merge_images(
                    to_original_size=to_original_size
                )
                image_fps.append(im_fp)

        if transform_non_reg:
            # preprocess and save unregistered nodes
            for modality in nonreg_keys:
                if modality in merge_modalities and remove_merged:
                    continue

                (
                    im_data,
                    transformations,
                    output_path,
                ) = self._prepare_nonreg_image_transform(
                    modality,
                    to_original_size=to_original_size,
                )
                im_fp = self._transform_write_image(
                    im_data,
                    transformations,
                    output_path,
                    file_writer=file_writer,
                )
                image_fps.append(im_fp)

        return image_fps

    def transform_shapes(self):
        """
        Transform all attached shapes and write out shape data to geoJSON.
        """
        transformed_shapes_fps = []

        # do inversion before any processing to ensure it is not computed twice
        for set_name, set_data in self.shape_sets.items():
            attachment_modality = set_data["attachment_modality"]
            if attachment_modality in self.transformations.keys():
                invert_nonrigid_transforms(
                    self.transformations[attachment_modality][
                        "full-transform-seq"
                    ].reg_transforms_itk_order
                )
            else:
                continue

        for set_name, set_data in self.shape_sets.items():
            attachment_modality = set_data["attachment_modality"]
            rs = RegShapes(
                set_data["shape_files"], source_res=set_data["image_res"]
            )

            if attachment_modality in self.transformations.keys():
                im_data = self.modalities[attachment_modality]
                final_modality = self.reg_paths[attachment_modality][-1]

                print(
                    "transforming shape set {} associated with {} to {}".format(
                        set_name, attachment_modality, final_modality
                    )
                )

                transformations = copy(
                    self.transformations[attachment_modality][
                        "full-transform-seq"
                    ]
                )

                if im_data.get("output_res"):
                    transformations.set_output_spacing(im_data["output_res"])

                else:
                    output_spacing_target = self.modalities[final_modality][
                        "image_res"
                    ]
                    transformations.set_output_spacing(
                        (output_spacing_target, output_spacing_target)
                    )

                self._transformed_shapes_spacings.update(
                    {set_name: (output_spacing_target, output_spacing_target)}
                )

                rs.transform_shapes(transformations)
                output_path = (
                    self.output_dir
                    / "{}-{}-{}_to_{}-transformed_shapes.geojson".format(
                        self.project_name,
                        set_name,
                        attachment_modality,
                        final_modality,
                    )
                )

                rs.save_shape_data(output_path, transformed=True)
                transformed_shapes_fps.append(output_path)
            else:
                output_path = (
                    self.output_dir
                    / "{}-{}-{}-registered.geojson".format(
                        self.project_name,
                        set_name,
                        attachment_modality,
                    )
                )

                output_spacing_target = self.modalities[attachment_modality][
                    "image_res"
                ]
                self._transformed_shapes_spacings.update(
                    {set_name: (output_spacing_target, output_spacing_target)}
                )
                rs.save_shape_data(output_path, transformed=False)
                transformed_shapes_fps.append(output_path)

        return transformed_shapes_fps

    def _transforms_to_txt(
        self, transformations: Dict[str, RegTransformSeq]
    ) -> Dict[str, List[str]]:
        tform_txt = dict()

        for k, v in transformations.items():
            if k == "full-transform-seq":
                continue
            if "initial" in k:
                if v:
                    for idx, rt in enumerate(v.reg_transforms):
                        tform_txt.update(
                            {f"{k}-{str(idx).zfill(2)}": rt.elastix_transform}
                        )
            else:
                tform_txt.update(
                    {f"{k}": [rt.elastix_transform for rt in v.reg_transforms]}
                )

        return tform_txt

    def save_transformations(self):
        """
        Save all transformations for a given modality as JSON
        """
        if all(
            [reg_edge.get("registered") for reg_edge in self.reg_graph_edges]
        ):
            for key in self.reg_paths.keys():
                final_modality = self.reg_paths[key][-1]

                output_path = (
                    self.output_dir
                    / "{}-{}_to_{}_transformations.json".format(
                        self.project_name,
                        key,
                        final_modality,
                    )
                )
                tform_txt = self._transforms_to_txt(self.transformations[key])

                with open(output_path, 'w') as fp:
                    json.dump(tform_txt, fp, indent=1)

            for (
                modality,
                attachment_modality,
            ) in self.attachment_images.items():
                if attachment_modality not in self._find_nonreg_modalities():
                    final_modality = self.reg_paths[attachment_modality][-1]

                    output_path = (
                        self.output_dir
                        / "{}-{}_to_{}_transformations.json".format(
                            self.project_name,
                            modality,
                            final_modality,
                        )
                    )

                    tform_txt = self._transforms_to_txt(
                        self.transformations[key]
                    )

                    with open(output_path, 'w') as fp:
                        json.dump(tform_txt, fp, indent=4)
        else:
            warn(
                "registration has not been executed for the graph "
                "no transformations to save"
            )

    def add_data_from_config(self, config_filepath):
        reg_config = parse_check_reg_config(config_filepath)

        if reg_config.get("modalities"):
            for key, val in reg_config["modalities"].items():
                image_filepath = (
                    val.get("image_filepath")
                    if val.get("image_filepath") is not None
                    else val.get("image_filepath")
                )

                preprocessing = (
                    "None"
                    if val.get("preprocessing") is None
                    else val.get("preprocessing")
                )

                self.add_modality(
                    key,
                    image_filepath,
                    image_res=val.get("image_res"),
                    channel_names=val.get("channel_names"),
                    channel_colors=val.get("channel_colors"),
                    preprocessing=preprocessing,
                    mask=val.get("mask"),
                    output_res=val.get("output_res"),
                )
        else:
            print("warning: config file did not contain any image modalities")

        if reg_config.get("reg_paths"):
            for key, val in reg_config["reg_paths"].items():
                self.add_reg_path(
                    val.get("src_modality_name"),
                    val.get("tgt_modality_name"),
                    val.get("thru_modality"),
                    reg_params=val.get("reg_params"),
                    override_prepro=val.get("override_prepro"),
                )
        else:
            print(
                "warning: config file did not contain any registration paths"
            )
        if reg_config.get("attachment_images"):
            for key, val in reg_config["attachment_images"].items():
                self.add_attachment_images(
                    val.get("attachment_modality"),
                    key,
                    val.get("image_filepath"),
                    val.get("image_res"),
                    channel_names=val.get("channel_names"),
                    channel_colors=val.get("channel_colors"),
                )

        if reg_config.get("attachment_shapes"):
            for key, val in reg_config["attachment_shapes"].items():
                self.add_attachment_shapes(
                    val.get("attachment_modality"), key, val.get("shape_files")
                )

        if reg_config.get("reg_graph_edges"):
            self._reg_graph_edges = reg_config["reg_graph_edges"]
            if all([re.get("registered") for re in self.reg_graph_edges]):
                self.transformations = self.reg_graph_edges

        if reg_config.get("merge_modalities"):
            for mn, mm in reg_config["merge_modalities"].items():
                self.add_merge_modalities(mn, mm)

    def reset_registered_modality(self, modalities):
        edge_keys = [
            r.get("modalities").get("source") for r in self.reg_graph_edges
        ]
        if isinstance(modalities, str):
            modalities = [modalities]

        for modality in modalities:
            modality_idx = edge_keys.index(modality)
            self.reg_graph_edges[modality_idx]["registered"] = False

    def _remove_modality_data(self, modality: str) -> None:
        if modality in self.modality_names:
            # remove top level
            self.modalities.pop(modality)
            self.modality_names.pop(self.modality_names.index(modality))

        to_rm = []
        for merge_name, merge_mods in self.merge_modalities.items():
            if modality in merge_mods:
                to_rm.append(merge_name)

        [self.merge_modalities.pop(k) for k in to_rm]

        self.n_modalities = len(self.modality_names)

        if modality in self.attachment_images.keys():
            self.attachment_images.pop(modality)

    def _remove_reg_paths(self, modality: str) -> None:
        # remove registrations
        if modality in self.reg_paths.keys():
            self.reg_paths.pop(modality)

        # remove any reg paths including modality as target or thru
        to_rm = []
        for k, v in self.reg_paths.items():
            if modality in v:
                to_rm.append(k)

        [self.reg_paths.pop(k) for k in to_rm]

        # remove any reg paths including modality as target or thru
        to_rm = []
        for k, v in self.transform_paths.items():
            for reg in v:
                if reg["source"] == modality or reg["target"] == modality:
                    to_rm.append(k)

        to_rm = list(np.unique(to_rm))

        edges_to_rm = []
        for idx, edge in enumerate(self.reg_graph_edges):
            if (
                edge["modalities"]["source"] in to_rm
                or edge["modalities"]["target"] in to_rm
            ):
                edges_to_rm.append(idx)

        [self.transform_paths.pop(k) for k in to_rm]

        for idx in sorted(edges_to_rm)[::-1]:
            self.reg_graph_edges.pop(idx)

        self.n_registrations = len(self._reg_graph_edges)

    def _remove_attachments(self, modality: str):
        to_rm = []
        for k, v in self.attachment_images.items():
            if modality == v:
                to_rm.append(k)

        [self.attachment_images.pop(k) for k in to_rm]
        [self.modalities.pop(k) for k in to_rm]
        [self.modality_names.pop(self.modality_names.index(k)) for k in to_rm]

        self.shape_sets.pop(modality)
        self.shape_set_names.pop(self.shape_set_names.index(modality))

        self.n_modalities = len(self.modality_names)

    def remove_merge_modality(self, merge_modality: str) -> None:
        try:
            self.merge_modalities.pop(merge_modality)
        except KeyError:
            warn(f"merge modality {merge_modality} not found")

    def remove_modality(self, modality: str) -> None:
        if modality in self.modality_names:
            self._remove_modality_data(modality)
            self._remove_reg_paths(modality)

        elif modality in self.shape_set_names:
            self._remove_attachments(modality)

        else:
            warn(
                f"{modality} not found in modalities: {', '.join(self.modality_names)}"
                f" or shape sets {self.shape_set_names}. "
                "It may have already been removed"
            )


def wsireg_run(
    graph_configuration: Union[str, Path, WsiReg2D],
    write_images: bool = True,
    to_original_size: bool = False,
    transform_non_reg: bool = True,
    remove_merged: bool = True,
    file_writer: str = "ome.tiff",
    testing: bool = False,
):
    def config_to_WsiReg2D(config_filepath):
        reg_config = parse_check_reg_config(config_filepath)

        reg_graph = WsiReg2D(
            reg_config.get("project_name"),
            reg_config.get("output_dir"),
            reg_config.get("cache_images"),
        )
        return reg_graph

    if isinstance(graph_configuration, (str, Path)):
        reg_graph = config_to_WsiReg2D(graph_configuration)
        reg_graph.add_data_from_config(graph_configuration)
    elif isinstance(graph_configuration, WsiReg2D):
        reg_graph = graph_configuration

    if testing:
        temp_dir = str(tempfile.mkdtemp())
        reg_graph.setup_project_output(reg_graph.project_name, temp_dir)

    reg_graph.register_images()
    reg_graph.save_transformations()
    output_data = []
    if write_images:
        output_images = reg_graph.transform_images(
            file_writer=file_writer,
            to_original_size=to_original_size,
            transform_non_reg=transform_non_reg,
            remove_merged=remove_merged,
        )
    else:
        output_images = []

    if reg_graph.shape_sets:
        output_shapes = reg_graph.transform_shapes()
    else:
        output_shapes = []

    output_data.extend(output_images)
    output_data.extend(output_shapes)

    if testing:
        shutil.rmtree(temp_dir)

    return output_data


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Load Whole Slide Image 2D Registration Graph from configuration file'
    )

    parser.add_argument(
        "config_filepath",
        metavar="C",
        type=str,
        help="full filepath for .yaml configuration file",
    )
    parser.add_argument(
        "--fw",
        type=str,
        help="how to write output registered images: ome.tiff, ome.zarr (default: ome.tiff)",
    )

    parser.add_argument('--write_im', dest='write_im', action='store_true')
    parser.add_argument('--no_write_im', dest='write_im', action='store_false')

    parser.add_argument(
        '--remove_merged', dest='remove_merged', action='store_true'
    )
    parser.add_argument(
        '--write_merge_indiv', dest='remove_merged', action='store_false'
    )

    parser.add_argument(
        '--tform_non_reg', dest='transform_non_reg', action='store_true'
    )
    parser.add_argument(
        '--no_tform_non_reg', dest='transform_non_reg', action='store_false'
    )

    parser.add_argument(
        '--to_orig_size', dest='to_original_size', action='store_true'
    )
    parser.add_argument(
        '--to_cropped', dest='to_original_size', action='store_false'
    )
    parser.add_argument('--testing', dest='testing', action='store_true')

    parser.set_defaults(
        write_im=True,
        remove_merged=True,
        transform_non_reg=True,
        to_original_size=False,
        testing=False,
    )

    args = parser.parse_args()
    config_filepath = args.config_filepath

    if args.fw is None:
        file_writer = "ome.tiff"
    else:
        file_writer = args.fw

    wsireg_run(
        config_filepath,
        write_images=args.write_im,
        to_original_size=args.to_original_size,
        transform_non_reg=args.transform_non_reg,
        remove_merged=args.remove_merged,
        file_writer=file_writer,
        testing=args.testing,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
