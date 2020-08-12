from pathlib import Path
import json
import SimpleITK as sitk
from wsireg.reg_utils import (
    RegImage,
    register_2d_images,
    apply_transform_dict,
    sitk_pmap_to_dict,
    pmap_dict_to_json,
    json_to_pmap_dict,
)
from wsireg.reg_shapes import RegShapes
from wsireg.im_utils import sitk_to_ometiff


class WsiReg2D(object):
    """
    Class to define a 2D registration graph and execute the registrations and transformations of the graph

    Parameters
    ----------
    project_name : str
        Project name will prefix all output files and directories
    output_dir : str
        Directory where registration data will be stored
    cache_images : bool
        whether to store images as they are preprocessed for registration (if you need to repeat or modify settings
        this will avoid image io and preprocessing)

    Attributes
    ----------
    modalities : dict
        dictionary of modality information (file path, spatial res., preprocessing), defines a graph node

    modalities : list
        list of all modality names

    n_modalities : int
        number of modalities (nodes) in the graphs

    reg_paths : dict
        dictionary of a modalities path from node to node

    reg_graph_edges : dict
        generated dictionary of necessary registrations to move modalities to their target modality

    n_registrations : int
        number of explicit registrations (edges) in the graphs

    transformation_paths : dict
        generated dictionary of necessary source - target transformations to transform modalities to their target modality

    transformations : dict
        per modality dictionary containing transformation parameters for each registration

    attachment_images : dict
        images to be transformed along the path of the defined graph, assoicated to a given modality (masks, other registered images)

    attachment_shapes : dict
        shape data attached to a modality to be transformed along the graph


    """

    def __init__(self, project_name: str, output_dir: str, cache_images=True):

        if project_name is None:
            self.project_name = 'RegProj'
        else:
            self.project_name = project_name

        self.output_dir = Path(output_dir)
        self.image_cache = self.output_dir / ".imcache_{}".format(project_name)
        self.cache_images = cache_images

        if self.cache_images is True:
            self.image_cache.mkdir(parents=False, exist_ok=True)

        self._modalities = {}
        self._modality_names = []
        self._reg_paths = {}
        self._reg_graph_edges = []
        self._transform_paths = {}

        self._transformations = None

        self.n_modalities = None
        self.n_registrations = None

        self.attachment_images = {}
        self.attachment_shapes = {}

        self._shape_sets = {}
        self._shape_set_names = []

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
                "image_fp": image_fp,
                "image_res": image_res,
                "channel_names": channel_names,
                "channel_colors": channel_colors,
                "preprocessing": prepro_dict,
            }
        }

        self.modality_names = modality_name

    def add_shape_set(
        self, shape_set_name, shape_files, image_res, attachment_modality
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

        self.shape_set = {
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
        if modality_name in self.modality_names:
            raise ValueError(
                'attachment modality named \"{}\" not found in modality_names'.format(
                    modality_name
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
        if attachment_modality in self.modality_names:
            raise ValueError(
                'attachment modality for shapes \"{}\" not found in modality_names'.format(
                    shape_set_name
                )
            )

        image_res = self.modalities[attachment_modality]["image_res"]
        self.add_shape_set(
            shape_set_name, shape_files, image_res, attachment_modality
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
            transform_edges = []
            for modality in tform_path:
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

        if cache_im_fp.exists() is True:
            im_fp = str(cache_im_fp)
            im_from_cache = True
        else:
            im_fp = self.modalities[modality_name]["image_fp"]
            im_from_cache = False

        if cache_transform_fp.exists() is True:
            im_initial_transforms = [json_to_pmap_dict(cache_transform_fp)]
        else:
            im_initial_transforms = None

        return im_fp, im_initial_transforms, im_from_cache

    def _prepare_modality(self, modality_name, reg_edge, src_or_tgt):
        mod_data = self.modalities[modality_name].copy()

        override_preprocessing = reg_edge.get("override_prepro")[src_or_tgt]

        if override_preprocessing is not None:
            mod_data["preprocessing"] = override_preprocessing

            return (
                mod_data["image_fp"],
                mod_data["image_res"],
                mod_data["preprocessing"],
                None,
            )
        else:

            (
                mod_data["image_fp"],
                mod_data["transforms"],
                im_from_cache,
            ) = self._check_cache_modality(modality_name)

            if im_from_cache is True:
                mod_data["preprocessing"] = None

            return (
                mod_data["image_fp"],
                mod_data["image_res"],
                mod_data["preprocessing"],
                mod_data["transforms"],
            )

    def _cache_images(self, modality_name, reg_image):

        cache_im_fp = self.image_cache / "{}_prepro.tiff".format(modality_name)

        cache_transform_fp = self.image_cache / "{}_init_tforms.json".format(
            cache_im_fp.stem
        )

        if cache_im_fp.is_file() is False:
            sitk.WriteImage(reg_image.image, str(cache_im_fp), True)

        if cache_transform_fp.is_file() is False:
            pmap_dict_to_json(reg_image.transforms, str(cache_transform_fp))

    def _find_nonreg_modalities(self):
        registered_modalities = [
            edge.get("modalities").get("source")
            for edge in self.reg_graph_edges
        ]
        return list(set(self.modality_names).difference(registered_modalities))

    def register_images(self, parallel=False, compute_inverse=False):
        """
        Start image registration process for all modalities

        Parameters
        ----------
        parallel : bool
            whether to run each edge in parallel (not implemented yet)
        compute_inverse : bool
            whether to compute the inverse transformation for each modality, may be used for point transformations but
            isn't currently working universally
        """
        for reg_edge in self.reg_graph_edges:

            src_name = reg_edge["modalities"]["source"]
            tgt_name = reg_edge["modalities"]["target"]

            (
                src_reg_image_fp,
                src_res,
                src_prepro,
                src_transforms,
            ) = self._prepare_modality(src_name, reg_edge, "source")

            (
                tgt_reg_image_fp,
                tgt_res,
                tgt_prepro,
                tgt_transforms,
            ) = self._prepare_modality(tgt_name, reg_edge, "target")

            src_reg_image = RegImage(
                src_reg_image_fp, src_res, src_prepro, src_transforms
            )

            tgt_reg_image = RegImage(
                tgt_reg_image_fp, tgt_res, tgt_prepro, tgt_transforms
            )

            if self.cache_images is True:
                if reg_edge["override_prepro"]["source"] is None:
                    self._cache_images(src_name, src_reg_image)
                if reg_edge["override_prepro"]["target"] is None:
                    self._cache_images(tgt_name, tgt_reg_image)

            reg_params = reg_edge["params"]

            output_path = self.output_dir / "{}-{}_to_{}_reg_output".format(
                self.project_name,
                reg_edge["modalities"]["source"],
                reg_edge["modalities"]["target"],
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

            reg_tforms = register_2d_images(
                src_reg_image,
                tgt_reg_image,
                reg_params,
                output_path,
                compute_inverse=compute_inverse,
            )

            reg_tforms = [sitk_pmap_to_dict(tf) for tf in reg_tforms]

            reg_edge["transforms"] = {
                'initial': src_reg_image.transforms,
                'registration': reg_tforms,
            }
            reg_edge["registered"] = True
            pmap_dict_to_json(reg_edge["transforms"], str(output_path_tform))

        self.transformations = self.reg_graph_edges

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, reg_graph_edges):
        self._transformations = self._get_transformations(reg_graph_edges)

    def _get_transformations(self, reg_graph_edges):
        transforms = {}
        edge_modality_pairs = [v['modalities'] for v in self.reg_graph_edges]
        for modality, tform_edges in self.transform_paths.items():
            for idx, tform_edge in enumerate(tform_edges):
                reg_edge_tforms = self.reg_graph_edges[
                    edge_modality_pairs.index(tform_edge)
                ]["transforms"]
                if idx == 0:
                    transforms[modality] = {
                        'initial': reg_edge_tforms['initial'],
                        idx: reg_edge_tforms['registration'],
                    }
                else:
                    transforms[modality][idx] = reg_edge_tforms['registration']

        return transforms

    def _transform_nonreg_image(self, modality_key, file_writer):
        print(
            "transforming non-registered modality : {} ".format(modality_key)
        )
        output_path = self.output_dir / "{}-{}_registered".format(
            self.project_name, modality_key
        )
        im_data = self.modalities[modality_key]

        if (
            im_data.get("preprocessing").get("rot_cc") is not None
            or im_data.get("preprocessing").get("flip") is not None
        ):
            transformations = {
                "initial": self._check_cache_modality(modality_key)[1][0],
                "registered": None,
            }
        else:
            transformations = None

        if file_writer == "sitk":
            image = apply_transform_dict(
                im_data["image_fp"], im_data["image_res"], transformations
            )
            sitk.WriteImage(image, str(output_path) + ".tiff", True)

        elif file_writer == "zarr":
            im_data_zarr = {
                "zarr_store_dir": str(output_path) + ".zarr",
                "channel_names": im_data["channel_names"],
                "channel_colors": im_data["channel_colors"],
            }

            zarr_image_fp = apply_transform_dict(
                im_data["image_fp"],
                im_data["image_res"],
                transformations,
                writer="zarr",
                **im_data_zarr,
            )

            return zarr_image_fp

    def _transform_image(
        self,
        edge_key,
        file_writer="sitk",
        attachment=False,
        attachment_modality=None,
    ):
        im_data = self.modalities[edge_key]

        if attachment is True:
            final_modality = self.reg_paths[attachment_modality][-1]
            transformations = self.transformations[attachment_modality]
        else:
            final_modality = self.reg_paths[edge_key][-1]
            transformations = self.transformations[edge_key]

        print("transforming {} to {}".format(edge_key, final_modality))

        output_path = self.output_dir / "{}-{}_to_{}_registered".format(
            self.project_name, edge_key, final_modality,
        )
        if file_writer == "sitk" or file_writer == "ometiff":
            image = apply_transform_dict(
                im_data["image_fp"], im_data["image_res"], transformations
            )
            if file_writer == "sitk":
                sitk.WriteImage(image, str(output_path) + ".tiff", True)
            elif file_writer == "ometiff":
                image_res = (image.GetSpacing()[0], image.GetSpacing()[1])
                image_name = "{}-{}_registered".format(
                    self.project_name, edge_key
                )
                sitk_to_ometiff(
                    image,
                    str(output_path) + "ome.tiff",
                    image_name=image_name,
                    channel_names=im_data["channel_names"],
                    image_res=image_res,
                )

        elif file_writer == "zarr":
            im_data_zarr = {
                "zarr_store_dir": str(output_path) + ".zarr",
                "channel_names": im_data["channel_names"],
                "channel_colors": im_data["channel_colors"],
            }

            zarr_image_fp = apply_transform_dict(
                im_data["image_fp"],
                im_data["image_res"],
                transformations,
                writer="zarr",
                **im_data_zarr,
            )

            return zarr_image_fp

    def transform_images(self, file_writer="sitk", transform_non_reg=True):
        """
        Transform and write images to disk after registration. Also transforms all attachment images

        Parameters
        ----------
        file_writer : str
            output type to use, sitk writes a single resolution tiff, "zarr" writes an ome-zarr multiscale
            zarr store
        """

        if file_writer == "zarr":
            zarr_paths = []

        if all(
            [reg_edge.get("registered") for reg_edge in self.reg_graph_edges]
        ):
            for key in self.reg_paths.keys():

                if file_writer == "zarr":
                    zarr_paths.append(
                        self._transform_image(key, file_writer=file_writer)
                    )
                elif file_writer == "sitk" or file_writer == "ometiff":
                    self._transform_image(key, file_writer=file_writer)

            for (
                modality,
                attachment_modality,
            ) in self.attachment_images.items():

                if file_writer == "zarr":
                    zarr_paths.append(
                        self._transform_image(
                            modality,
                            file_writer=file_writer,
                            attachment=True,
                            attachment_modality=attachment_modality,
                        )
                    )
                elif file_writer == "sitk" or file_writer == "ometiff":
                    self._transform_image(
                        modality,
                        file_writer=file_writer,
                        attachment=True,
                        attachment_modality=attachment_modality,
                    )
            else:
                print(
                    "warning: not all edges have been registered, skipping transformation of registered images"
                )

        if transform_non_reg is True:
            # preprocess and save unregistered nodes
            nonreg_keys = self._find_nonreg_modalities()

            for key in nonreg_keys:
                zarr_paths.append(
                    self._transform_nonreg_image(key, file_writer=file_writer)
                )
        return zarr_paths

    def transform_shapes(self):
        """
        Transform all attached shapes and write out shape data to geoJSON.
        """
        for k, v in self.shape_sets:
            set_data = v
            attachment_modality = self.shape_sets["attachment_modality"]

            final_modality = self.reg_paths[attachment_modality][-1]

            print(
                "transforming shapes associated with {} to {}".format(
                    attachment_modality, final_modality
                )
            )

            rs = RegShapes(set_data["shape_files"])
            rs.transform_shapes(
                set_data["image_res"],
                self.transformations[attachment_modality],
            )

            output_path = (
                self.output_dir
                / "{}-{}_to_{}_transformed_shapes.json".format(
                    self.project_name, attachment_modality, final_modality
                )
            )

            rs.save_shape_data(output_path)

    def save_transformations(self):
        """
        Save all transformations for a given modality as JSON
        """
        for key in self.reg_paths.keys():

            final_modality = self.reg_paths[key][-1]

            output_path = (
                self.output_dir
                / "{}-{}_to_{}_transformations.json".format(
                    self.project_name, key, final_modality,
                )
            )

            with open(output_path, 'w') as fp:
                json.dump(self.transformations[key], fp, indent=4)

        for (modality, attachment_modality,) in self.attachment_images.items():

            final_modality = self.reg_paths[attachment_modality][-1]

            output_path = (
                self.output_dir
                / "{}-{}_to_{}_transformations.json".format(
                    self.project_name, modality, final_modality,
                )
            )

            with open(output_path, 'w') as fp:
                json.dump(self.transformations[key], fp, indent=4)


# TODO: add command line control & config files
