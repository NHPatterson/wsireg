# wsireg
Multi-modal or mono-modal whole slide image registration in a graph structure for complex registration tasks using `elastix`.

## Features
* Graph based approach to defining modalities and arbitrary transformation paths between associated images
* Use of `elastix` (through [ITKElastix](https://github.com/InsightSoftwareConsortium/ITKElastix/)) to perform registration
* Support for linear and non-linear transformation models
* Transform associated data (masks, shape data) along the same path as the images.
* Supports images converted to OME-TIFF using [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) -> [raw2ometiff](https://github.com/glencoesoftware/bioformats2raw) pipeline
* All registered images exported as pyramidal OME-TIFF or OME-zarr that can be viewed in software such as [Vitessce](vitessce.io),[vizarr](https://github.com/hms-dbmi/vizarr), [QuPath](https://qupath.github.io), [OMERO](https://www.openmicroscopy.org/omero/) or any platform that supports these formats.
* All transforms for complex registration paths are internally composited and only 1 interpolation step is performed, avoiding accumulation of interpolation error from many registrations
* Shape data (polygons, point sets, etc.) in GeoJSON format (future portable format for QuPath detection/annotation data) can be imported and transformations applied producing a modified GeoJSON
* Some support for reading native WSI formats: currently reads .czi and .scn but could be expanded to other formats supported by `tifffile`


## Installation
Install cross-platform Python packages with [pip](https://pypi.org/project/pip/):

```bash
pip install wsireg
```

## Usage
### Python usage
Example registering two images
```python
from wsireg.wsireg2d import WsiReg2D

# initialize registration graph
reg_graph = WsiReg2D("my_reg_project", "./project_folder")

# add registration images (modalities)
reg_graph.add_modality(
    "modality_fluo",
    "./data/im1.tiff",
    image_res=0.65,
    channel_names=["DAPI", "eGFP", "DsRed"],
    channel_colors=["blue", "green", "red"],
    prepro_dict={
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
    prepro_dict={"image_type": "BF", "as_uint8": True, "inv_int_opt": True},
)

reg_graph.add_reg_path(
    "modality_fluo",
    "modality_brightfield",
    thru_modality=None,
    reg_params=["rigid", "affine"],
)

reg_graph.register_images()
reg_graph.save_transformations()
reg_graph.transform_images(file_writer="ome.tiff")
```
This will register the images and save them in the project output directory as tiled, pyramidal OME-TIFF. Original data types are respected for images.
### Command line with .yaml configuration

## Future support
* Complete support for masking and cropping based on masks
* 3D registration
* Automatic image conversion using `bioformats`
* Use of registration libraries besides `elastix`
* Memory-efficient transformation
* Deeper OME metadata

