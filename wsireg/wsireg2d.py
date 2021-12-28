import time
import yaml
from pathlib import Path
from copy import deepcopy
import json
import SimpleITK as sitk
from wsireg.reg_images.loader import reg_image_loader
from wsireg.reg_images import MergeRegImage
from wsireg.writers.merge_ome_tiff_writer import OmeTiffWriter
from wsireg.writers.merge_ome_tiff_writer import MergeOmeTiffWriter
from wsireg.utils.reg_utils import (
    register_2d_images_itkelx,
    sitk_pmap_to_dict,
    pmap_dict_to_json,
    json_to_pmap_dict,
)
from wsireg.reg_shapes import RegShapes
from wsireg.reg_transform import RegTransform
from wsireg.utils.config_utils import parse_check_reg_config
from wsireg.utils.tform_conversion import get_elastix_transforms
from wsireg.utils.tform_utils import (
    prepare_wsireg_transform_data,
)
from wsireg.utils.im_utils import (
    ARRAYLIKE_CLASSES,
)


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

    Attributes
    ----------
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


    """

    def __init__(
        self,
        project_name: str,
        output_dir: str,
        cache_images=True,
        config=None,
    ):

        if project_name is None:
            self.project_name = 'RegProj'
        else:
            self.project_name = project_name

        if output_dir is None:
            output_dir = "./"
        self.output_dir = Path(output_dir)
        self.image_cache = self.output_dir / ".imcache_{}".format(project_name)
        self.cache_images = cache_images

        self.pairwise = False

        self._modalities = {}
        self._modality_names = []
        self._reg_paths = {}
        self._reg_graph_edges = []
        self._transform_paths = {}

        self._transformations = None

        self.n_modalities = None
        self.n_registrations = None

        self.attachment_images = {}

        self._shape_sets = {}
        self._shape_set_names = []

        self.merge_modalities = {}
        self.original_size_transforms = {}

        if config is not None:
            self.add_data_from_config(config)

    @property
    def modalities(self):
        return self._modalities

    @modalities.setter
    def modalities(self, modality):
        self._modalities.update(modality)
        self.n_modalities = len(self._modalities)

    @property
    def shape_sets(self):
        return self._shape_sets

    @shape_sets.setter
    def shape_sets(self, shape_set):
        self._shape_sets.update(shape_set)

    @property
    def shape_set_names(self):
        return self._shape_set_names

    @shape_set_names.setter
    def shape_set_names(self, shape_set_name):
        self._shape_set_names.append(shape_set_name)

    @property
    def modality_names(self):
        return self._modality_names

    @modality_names.setter
    def modality_names(self, modality_name):
        self._modality_names.append(modality_name)

    def add_modality(
        self,
        modality_name,
        image_fp,
        image_res=1,
        channel_names=None,
        channel_colors=None,
        prepro_dict={},
        mask=None,
    ):
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
        prepro_dict :
            preprocessing parameters for the modality for registration. Registration images should be a xy single plane
            so many modalities (multi-channel, RGB) must "create" a single channel.
            Defaults: multi-channel images -> max intensity project image
            RGB -> greyscale then intensity inversion (black background, white foreground)
        """
        if modality_name in self._modality_names:
            raise ValueError(
                'modality named \"{}\" is already in modality_names'.format(
                    modality_name
                )
            )

        self.modalities = {
            modality_name: {
                "image_filepath": image_fp,
                "image_res": image_res,
                "channel_names": channel_names,
                "channel_colors": channel_colors,
                "preprocessing": prepro_dict,
                "mask": mask,
            }
        }

        self.modality_names = modality_name

    def add_shape_set(
        self, attachment_modality, shape_set_name, shape_files, image_res
    ):
        """
        Add a shape set to the graph

        Parameters
        ----------
        shape_set_name : str
            Unique name identifier for the shape set
        shape_files : list
            list of shape data in geoJSON format or list of dicts containing following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        image_res : float
            spatial resolution of shape data's associated image in units per px (i.e. 0.9 um / px)
        attachment_modality :
            image modality to which the shapes are attached
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
        attachment_modality,
        modality_name,
        image_fp,
        image_res=1,
        channel_names=None,
        channel_colors=None,
    ):
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
        self, attachment_modality, shape_set_name, shape_files
    ):
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
    def reg_paths(self):
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

        self.reg_graph_edges = {
            'modalities': {'source': src_modality, 'target': thru_modality},
            'params': reg_params,
            "override_prepro": override_prepro,
        }
        self.transform_paths = self._reg_paths

    def add_reg_path(
        self,
        src_modality_name,
        tgt_modality_name,
        thru_modality=None,
        reg_params=[],
        override_prepro={"source": None, "target": None},
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
        reg_params:
            SimpleElastix registration parameters
        override_prepro:
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

    def find_path(self, start_modality, end_modality, path=None):
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

    def _check_cache_modality(self, modality_name):
        cache_im_fp = self.image_cache / "{}_prepro.tiff".format(modality_name)
        cache_transform_fp = cache_im_fp.parent / "{}_init_tforms.json".format(
            cache_im_fp.stem
        )
        cache_osize_tform_fp = (
            self.image_cache
            / "{}_orig_size_tform.json".format(cache_im_fp.stem)
        )

        if cache_im_fp.exists() is True:
            im_fp = str(cache_im_fp)
            im_from_cache = True
        else:
            im_fp = self.modalities[modality_name]["image_filepath"]
            im_from_cache = False

        if cache_transform_fp.exists() is True:
            im_initial_transforms = [json_to_pmap_dict(cache_transform_fp)]
        else:
            im_initial_transforms = None

        if cache_osize_tform_fp.exists() is True:
            osize_tform = json_to_pmap_dict(cache_osize_tform_fp)
        else:
            osize_tform = None

        return im_fp, im_initial_transforms, im_from_cache, osize_tform

    def _prepare_modality(self, modality_name, reg_edge, src_or_tgt):
        mod_data = self.modalities[modality_name].copy()

        if reg_edge.get("override_prepro") is not None:
            override_preprocessing = reg_edge.get("override_prepro")[
                src_or_tgt
            ]
        else:
            override_preprocessing = None

        if override_preprocessing is not None:
            mod_data["preprocessing"] = override_preprocessing

            return (
                mod_data["image_filepath"],
                mod_data["image_res"],
                mod_data["preprocessing"],
                None,
                mod_data["mask"],
                None,
            )
        else:

            (
                mod_data["image_filepath"],
                mod_data["transforms"],
                im_from_cache,
                original_size_transform,
            ) = self._check_cache_modality(modality_name)

            if im_from_cache is True:
                if mod_data["preprocessing"].get("use_mask") is False:
                    mod_data["mask"] = None
                mod_data["preprocessing"] = None

            return (
                mod_data["image_filepath"],
                mod_data["image_res"],
                mod_data["preprocessing"],
                mod_data["transforms"],
                mod_data["mask"],
                original_size_transform,
            )

    def _cache_images(self, modality_name, reg_image):

        cache_im_fp = self.image_cache / "{}_prepro.tiff".format(modality_name)

        cache_transform_fp = self.image_cache / "{}_init_tforms.json".format(
            cache_im_fp.stem
        )

        cache_osize_tform_fp = (
            self.image_cache
            / "{}_orig_size_tform.json".format(cache_im_fp.stem)
        )
        if cache_im_fp.is_file() is False:
            sitk.WriteImage(reg_image.reg_image, str(cache_im_fp), True)

        if reg_image.mask is not None:
            cache_mask_im_fp = self.image_cache / "{}_prepro_mask.tiff".format(
                modality_name
            )
            if cache_mask_im_fp.is_file() is False:
                sitk.WriteImage(reg_image.mask, str(cache_mask_im_fp), True)

        if cache_transform_fp.is_file() is False:
            pmap_dict_to_json(
                reg_image.pre_reg_transforms, str(cache_transform_fp)
            )

        if (
            cache_osize_tform_fp.is_file() is False
            and reg_image.original_size_transform is not None
        ):
            pmap_dict_to_json(
                reg_image.original_size_transform, str(cache_osize_tform_fp)
            )

    def _find_nonreg_modalities(self):
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

    def save_config(self, registered=False):
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

        [rge.pop("reg_transforms", None) for rge in reg_graph_edges]

        modalities_out = deepcopy(self.modalities)
        for mod, data in modalities_out.items():
            if isinstance(data["image_filepath"], ARRAYLIKE_CLASSES):
                data["image_filepath"] = "ArrayLike"

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
        }

        output_path = (
            self.output_dir
            / f"{ts}-{self.project_name}-configuration-{status}.yaml"
        )

        with open(str(output_path), "w") as f:
            yaml.dump(config, f, sort_keys=False)

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

                (
                    src_reg_image_fp,
                    src_res,
                    src_prepro,
                    src_transforms,
                    src_mask,
                    src_original_size_transform,
                ) = self._prepare_modality(src_name, reg_edge, "source")

                (
                    tgt_reg_image_fp,
                    tgt_res,
                    tgt_prepro,
                    tgt_transforms,
                    tgt_mask,
                    tgt_original_size_transform,
                ) = self._prepare_modality(tgt_name, reg_edge, "target")

                src_reg_image = reg_image_loader(
                    src_reg_image_fp,
                    src_res,
                    preprocessing=src_prepro,
                    pre_reg_transforms=src_transforms,
                    mask=src_mask,
                )

                tgt_reg_image = reg_image_loader(
                    tgt_reg_image_fp,
                    tgt_res,
                    preprocessing=tgt_prepro,
                    pre_reg_transforms=tgt_transforms,
                    mask=tgt_mask,
                )

                src_reg_image.read_reg_image()
                tgt_reg_image.read_reg_image()

                if (
                    tgt_original_size_transform is None
                    and tgt_reg_image.original_size_transform is not None
                ):
                    tgt_original_size_transform = (
                        tgt_reg_image.original_size_transform
                    )

                if self.cache_images is True:
                    if reg_edge.get("override_prepro") is not None:
                        if (
                            reg_edge.get("override_prepro").get("source")
                            is None
                        ):
                            self._cache_images(src_name, src_reg_image)
                        if (
                            reg_edge.get("override_prepro").get("target")
                            is None
                        ):
                            self._cache_images(tgt_name, tgt_reg_image)
                    else:
                        self._cache_images(src_name, src_reg_image)
                        self._cache_images(tgt_name, tgt_reg_image)

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

                output_path_tform = (
                    self.output_dir
                    / "{}-{}_to_{}_transformations.json".format(
                        self.project_name,
                        reg_edge["modalities"]["source"],
                        reg_edge["modalities"]["target"],
                    )
                )

                reg_tforms = register_2d_images_itkelx(
                    src_reg_image,
                    tgt_reg_image,
                    reg_params,
                    output_path,
                )

                reg_tforms = [sitk_pmap_to_dict(tf) for tf in reg_tforms]

                if src_transforms is not None:
                    initial_transforms = src_transforms[0]
                else:
                    initial_transforms = src_reg_image.pre_reg_transforms

                reg_edge["transforms"] = {
                    'initial': initial_transforms,
                    'registration': reg_tforms,
                }

                self.original_size_transforms.update(
                    {tgt_name: tgt_original_size_transform}
                )

                reg_edge["registered"] = True
                pmap_dict_to_json(
                    reg_edge["transforms"], str(output_path_tform)
                )

        self.transformations = self.reg_graph_edges
        self.save_config(registered=True)

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
        transforms = {}
        for reg_edge in self.reg_graph_edges:
            if reg_edge["transforms"]["initial"] is not None:
                initial_transforms = [
                    RegTransform(t) for t in reg_edge["transforms"]["initial"]
                ]
            else:
                initial_transforms = []
            reg_edge["reg_transforms"] = {
                'initial': initial_transforms,
                'registration': [
                    RegTransform(t)
                    for t in reg_edge["transforms"]["registration"]
                ],
            }
        edge_modality_pairs = [v['modalities'] for v in self.reg_graph_edges]
        for modality, tform_edges in self.transform_paths.items():
            for idx, tform_edge in enumerate(tform_edges):
                reg_edge_tforms = self.reg_graph_edges[
                    edge_modality_pairs.index(tform_edge)
                ]["reg_transforms"]
                if idx == 0:
                    transforms[modality] = {
                        'initial': reg_edge_tforms['initial'],
                        idx: reg_edge_tforms['registration'],
                    }
                else:
                    transforms[modality][idx] = reg_edge_tforms['registration']

        return transforms

    def _prepare_nonreg_image_transform(
        self, modality_key, to_original_size=True
    ):
        print(
            "transforming non-registered modality : {} ".format(modality_key)
        )
        output_path = self.output_dir / "{}-{}_registered".format(
            self.project_name, modality_key
        )
        im_data = self.modalities[modality_key]
        transformations = {"initial": None, "registered": None}

        if (
            im_data.get("preprocessing").get("rot_cc") is not None
            or im_data.get("preprocessing").get("flip") is not None
            or im_data.get("preprocessing").get("mask_to_bbox") is True
            or im_data.get("preprocessing").get("mask_bbox") is not None
        ):

            transformations.update(
                {
                    "initial": self._check_cache_modality(modality_key)[1][0],
                    "registered": None,
                }
            )

        if to_original_size is True:
            transformations.update(
                {"registered": self.original_size_transforms.get(modality_key)}
            )

        if (
            transformations.get("initial") is None
            and transformations.get("registered") is None
        ):
            transformations = None

        return im_data, transformations, output_path

    def _prepare_reg_image_transform(
        self,
        edge_key,
        attachment=False,
        attachment_modality=None,
        to_original_size=True,
    ):
        im_data = self.modalities[edge_key]

        if attachment is True:
            final_modality = self.reg_paths[attachment_modality][-1]
            transformations = deepcopy(
                self.transformations[attachment_modality]
            )
        else:
            final_modality = self.reg_paths[edge_key][-1]
            transformations = deepcopy(self.transformations[edge_key])

        print("transforming {} to {}".format(edge_key, final_modality))

        output_path = self.output_dir / "{}-{}_to_{}_registered".format(
            self.project_name,
            edge_key,
            final_modality,
        )
        if (
            self.original_size_transforms.get(final_modality) is not None
            and to_original_size is True
        ):
            original_size_transform = self.original_size_transforms[
                final_modality
            ]
            transformations.update(
                {"orig": [RegTransform(original_size_transform)]}
            )

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

        ometiffwriter = OmeTiffWriter(tfregimage)

        if transformations:
            (
                composite_transform,
                itk_transforms,
                final_transform,
            ) = prepare_wsireg_transform_data(transformations)
        else:
            composite_transform, itk_transforms, final_transform = (
                None,
                None,
                None,
            )

        if (
            file_writer == "ome.tiff-bytile"
            and ometiffwriter.reg_image.reader not in ["czi", "sitk"]
        ):
            im_fp = ometiffwriter.write_image_by_tile(
                output_path.stem,
                itk_transforms=itk_transforms,
                composite_transform=composite_transform,
                final_transform=final_transform,
                output_dir=str(self.output_dir),
            )
        else:
            im_fp = ometiffwriter.write_image_by_plane(
                output_path.stem,
                composite_transform=composite_transform,
                final_transform=final_transform,
                output_dir=str(self.output_dir),
            )

        return im_fp

    def _transform_write_merge_images(self, to_original_size=True):
        for merge_name, sub_images in self.merge_modalities.items():
            im_fps = []
            im_res = []
            im_ch_names = []
            transformations = []
            final_modalities = []
            for sub_image in sub_images:
                im_data = self.modalities[sub_image]
                im_fps.append(im_data["image_filepath"])
                im_res.append(im_data["image_res"])
                im_ch_names.append(im_data.get("channel_names"))

                try:
                    transforms = deepcopy(self.transformations[sub_image])
                except KeyError:
                    transforms = None

                try:
                    final_modalities.append(self.reg_paths[sub_image][-1])
                except KeyError:
                    initial_transforms = self._check_cache_modality(sub_image)[
                        1
                    ][0]
                    if initial_transforms:
                        initial_transforms = [
                            RegTransform(t) for t in initial_transforms
                        ]
                        final_modalities.append(sub_image)
                        transforms = {"initial": initial_transforms}
                    else:
                        transforms = None

                transformations.append(transforms)

            if all(final_modalities):
                final_modality = final_modalities[0]
            else:
                raise ValueError("final modalities do not match on merge")

            if (
                self.original_size_transforms.get(final_modality) is not None
                and to_original_size is True
            ):
                original_size_transform = self.original_size_transforms[
                    final_modality
                ]
                for transformation in transformations:
                    if transformation is None:
                        transformation = {}
                    transformation.update(
                        {"orig": [RegTransform(original_size_transform)]}
                    )

            output_path = self.output_dir / "{}-{}_merged-registered".format(
                self.project_name,
                merge_name,
            )

            merge_regimage = MergeRegImage(
                im_fps,
                im_res,
                channel_names=im_ch_names,
            )

            merge_ometiffwriter = MergeOmeTiffWriter(merge_regimage)

            im_fp = merge_ometiffwriter.merge_write_image_by_plane(
                output_path.stem,
                sub_images,
                transformations=transformations,
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

        if transform_non_reg is True:
            # preprocess and save unregistered nodes
            for modality in nonreg_keys:
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
        for set_name, set_data in self.shape_sets.items():
            attachment_modality = set_data["attachment_modality"]

            final_modality = self.reg_paths[attachment_modality][-1]

            print(
                "transforming shape set {} associated with {} to {}".format(
                    set_name, attachment_modality, final_modality
                )
            )

            rs = RegShapes(
                set_data["shape_files"], source_res=set_data["image_res"]
            )
            rs.transform_shapes(
                self.transformations[attachment_modality],
            )

            output_path = (
                self.output_dir
                / "{}-{}-{}_to_{}-transformed_shapes.json".format(
                    self.project_name,
                    set_name,
                    attachment_modality,
                    final_modality,
                )
            )

            rs.save_shape_data(output_path, transformed=True)

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
                out_transforms = get_elastix_transforms(
                    self.transformations[key]
                )

                with open(output_path, 'w') as fp:
                    json.dump(out_transforms, fp, indent=4)

            for (
                modality,
                attachment_modality,
            ) in self.attachment_images.items():

                final_modality = self.reg_paths[attachment_modality][-1]

                output_path = (
                    self.output_dir
                    / "{}-{}_to_{}_transformations.geojson".format(
                        self.project_name,
                        modality,
                        final_modality,
                    )
                )

                out_transforms = get_elastix_transforms(
                    self.transformations[attachment_modality]
                )

                with open(output_path, 'w') as fp:
                    json.dump(out_transforms, fp, indent=4)
        else:
            print("registration has not been executed for the graph")

    def add_data_from_config(self, config_filepath):

        reg_config = parse_check_reg_config(config_filepath)

        if reg_config.get("modalities") is not None:
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
                    prepro_dict=preprocessing,
                    mask=val.get("mask"),
                )
        else:
            print("warning: config file did not contain any image modalities")

        if reg_config.get("reg_paths") is not None:

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
        if reg_config.get("attachment_images") is not None:

            for key, val in reg_config["attachment_images"].items():
                self.add_attachment_images(
                    val.get("attachment_modality"),
                    key,
                    val.get("image_filepath"),
                    val.get("image_res"),
                    channel_names=val.get("channel_names"),
                    channel_colors=val.get("channel_colors"),
                )

        if reg_config.get("attachment_shapes") is not None:

            for key, val in reg_config["attachment_shapes"].items():
                self.add_attachment_shapes(
                    val.get("attachment_modality"), key, val.get("shape_files")
                )

        if reg_config.get("reg_graph_edges") is not None:
            self._reg_graph_edges = reg_config["reg_graph_edges"]
            if all([re.get("registered") for re in self.reg_graph_edges]):
                self.transformations = self.reg_graph_edges

        if reg_config.get("merge_modalities") is not None:
            for mn, mm in reg_config["merge_modalities"]:
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


def main():
    import argparse

    def config_to_WsiReg2D(config_filepath):
        reg_config = parse_check_reg_config(config_filepath)

        reg_graph = WsiReg2D(
            reg_config.get("project_name"),
            reg_config.get("output_dir"),
            reg_config.get("cache_images"),
        )
        return reg_graph

    parser = argparse.ArgumentParser(
        description='Load Whole Slide Image 2D Registration Graph from configuration file'
    )

    parser.add_argument(
        "config_filepath",
        metavar="C",
        type=str,
        nargs=1,
        help="full filepath for .yaml configuration file",
    )
    parser.add_argument(
        "--fw",
        type=str,
        nargs=1,
        help="how to write output registered images: ome.tiff, ome.zarr (default: ome.tiff)",
    )

    args = parser.parse_args()
    config_filepath = args.config_filepath[0]
    if args.fw is None:
        file_writer = "ome.tiff"
    else:
        file_writer = args.fw[0]

    reg_graph = config_to_WsiReg2D(config_filepath)
    reg_graph.add_data_from_config(config_filepath)

    reg_graph.register_images()
    reg_graph.save_transformations()
    reg_graph.transform_images(file_writer=file_writer)

    if reg_graph.shape_sets:
        reg_graph.transform_shapes()


if __name__ == "__main__":
    import sys

    sys.exit(main())
