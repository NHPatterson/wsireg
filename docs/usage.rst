=====
Usage
=====


The general workflow for setting up a run is this, below we show this as well as use this as a base to explore
additional wsireg features.

#. Initialize graph with name and an output directory using WsiReg2D.

#. Add images using add_modality.

#. Add registration paths using add_reg_path, setting appropriate moving and fixed modality names and
   the types of registration parameters that should be used for alignment of this particular pair.

#. Run graph by using .register_images().

#. Save transformations using .save_transformations(). This writes transformation .json files to the project directory.

#. Transform and resample images using .transform_images(). This writes OME-TIFF mages to the project directory.

Simple two image example
#########################

In this example, two images are aligned, "modality_fluo" is aligned to "modality_brightfield".


.. code-block:: python

    from wsireg import WsiReg2D

    # initialize registration graph
    reg_graph = WsiReg2D("my_reg_project", "./project_folder")

    # add registration images (modalities)
    reg_graph.add_modality(
        "modality_fluo",
        "./data/im1.tiff",
        image_res=0.65,
        channel_names=["DAPI", "eGFP", "DsRed"],
        channel_colors=["blue", "green", "red"],
        preprocessing={
            "image_type": "FL",
            "ch_indices": [1],
            "as_uint8": True,
            "contrast_enhance": True,
        },
    )

    reg_graph.add_modality(
        "modality_brightfield",
        "./data/im2.tiff",
        image_res=1,
        preprocessing={
            "image_type": "BF",
            "as_uint8": True,
            "invert_intensity": True,
        },
    )

    # we register here the fluorescence modality to the brightfield
    # using a rigid and affine parameter map
    reg_graph.add_reg_path(
        "modality_fluo",
        "modality_brightfield",
        thru_modality=None,
        reg_params=["rigid", "affine"],
    )

    # run the graph
    reg_graph.register_images()

    # save transformation data
    reg_graph.save_transformations()

    # save registerd images as ome.tiff writing images
    # plane by plane
    reg_graph.transform_images(file_writer="ome.tiff")

Transform shapes in geojson file exported from QuPath using WsiReg2D
#####################################################################

Shapes in wsireg are a layer that includes polygons, points, line strings or other shape
types. In wsireg, shapes maps the concept of the `attachment shapes`. These are sets of coordinates that form
polygons, points, line strings, etc. that are associated with a given modality. For now, only geojson files are supported
as an input type (export from QuPath >0.3.2).

Point sets are expect to be in pixel coordinates (of the base resolution of the WSI) rather than physical coordinates.
As all elastix/ITK operations work in physical coordinate space, the coordinates are transformed to their physical locations
using the image spacing of the image to which the shape data is attached.

Within QuPath, shapes can be exported as a geojson from the File menu. If data is generated
from another source, saving them in geojson from numpy for use in wsireg is possible with the geojson python library.

Within elastix, transforms are mapped from fixed space to moving space. A simple way to think of this
is that for each pixel in the fixed image, we find the best matching pixel in the moving image. Unfortunately,
this approach does not produce a transformation that is appropriate for transforming sets of coordinates in moving
space to the fixed space, thus the inverse transformation (transforms coordinates FROM moving TO fixed) is used.

For linear transforms, this is simply the inverse of the transformation matrix, but for non-linear transformations,
the inverse must be estimated which works almost losslessly in 2D scenarios but can consume a lot of memory during genration.

This explanation is intended for general knowledge. Within wsireg, all of these operations are
performed without user input.

Below we demonstrate adding shapes to the registration graph starting from the above model after
the image modalities have been added:

.. code-block:: python

    # add attachment shapes here
    # first arg: previously defined image modality
    # to which shapes are attached
    # second: name for shapes (propgates to file output)
    # third: path to data
    reg_graph.add_attachment_shapes(
        "modality_fluo", "fluo_annotations", "./data/fluo_annotations.geojson"
    )
    # same as previous
    reg_graph.add_reg_path(
        "modality_fluo",
        "modality_brightfield",
        thru_modality=None,
        reg_params=["rigid", "affine"],
    )

    # same as previous
    reg_graph.register_images()
    reg_graph.save_transformations()
    reg_graph.transform_images(file_writer="ome.tiff")

    # adding this line runs the shape transformation and saves the transforms as
    # geojson in the project output directory
    reg_graph.transform_shapes()

Transform derived images associated with a modality using WsiReg2D
##################################################################

In some scenarios, a derived image has been made for a given modality. For instance, a binary or label map image
of cells or other spatial or spatial/intensity features in the WSI.
It may be desriable to transform and resample this image to the coordinate space of the target modality allowing
mapping of a dervied feature from one image onto another for comparison.

In wsireg, this maps the concept of the `attachment image`. Much like `attachment shapes`, these are images associated
to a given modality that can be transformed on the same registration path as that modality.

Starting from the previous example, only one additional line of code is needed:

.. code-block:: python

    # add a cell segmentation mask to the fluorescence modality
    reg_graph.add_attachment_images(
        "modality_fluo",
        "fluo_cell_seg",
        "./data/im1_segmentation.tiff",
        0.65,
        channel_names=["nuclei mask", "cell boundary"],
    )

    # we register here the fluorescence modality to the brightfield
    # using a rigid and affine parameter map
    reg_graph.add_reg_path(
        "modality_fluo",
        "modality_brightfield",
        thru_modality=None,
        reg_params=["rigid", "affine"],
    )

    # run the graph
    reg_graph.register_images()

    # save transformation data
    reg_graph.save_transformations()

    # save registerd images as ome.tiff writing images
    # plane by plane
    reg_graph.transform_images(file_writer="ome.tiff")

Changing registration parameters (rigid, affine, non-linear, etc.)
###################################################################

In the above example we use a parameter maps by name, parameter maps can be accessed
programmatically as well as shown below. Using the example set up above, here the
registration model is changed to rigid, affine and non-rigid transformations in sequence.
This works well as the rigid transform finds a good starting point prior to affine (scaling/shearing).

.. code-block:: python

    # we register here the fluorescence modality to the brightfield
    # using a rigid and affine AND an non-rigid parameter
    # accessed using the RegModel enum
    # in an IDE this will populate with all the various options
    # alterntatively if you add a string file path that
    # goes to an elastix
    reg_graph.add_reg_path(
        "modality_fluo",
        "modality_brightfield",
        thru_modality=None,
        reg_params=[RegModel.rigid, RegModel.affine, RegModel.nl],
    )
