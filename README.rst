======
wsireg
======


.. image:: https://readthedocs.org/projects/wsireg/badge/?version=latest
        :target: https://wsireg.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Multi-modal or mono-modal whole slide image registration in a graph structure for complex registration tasks using `elastix`.


* Documentation: https://wsireg.readthedocs.io.


Features
--------

* Graph based approach to defining modalities and arbitrary transformation paths between associated images
* Use of `elastix` (through `ITKElastix <https://github.com/InsightSoftwareConsortium/ITKElastix/>`_) to perform registration
* Support for linear and non-linear transformation models
* Transform associated data (masks, shape data) along the same path as the images.
* Supports images converted to OME-TIFF using `bioformats2raw <https://github.com/glencoesoftware/bioformats2raw>`_ -> `raw2ometiff <https://github.com/glencoesoftware/bioformats2raw>`_ pipeline
* All registered images exported as pyramidal OME-TIFF or OME-zarr that can be viewed in software such as `Vitessce <https://vitessce.io>`_ , `vizarr <https://github.com/hms-dbmi/vizarr>`_, `QuPath <https://qupath.github.io>`_, `OMERO <https://www.openmicroscopy.org/omero/>`_ or any platform that supports these formats.
* All transforms for complex registration paths are internally composited and only 1 interpolation step is performed, avoiding accumulation of interpolation error from many registrations
* Shape data (polygons, point sets, etc.) in GeoJSON format (future portable format for QuPath detection/annotation data) can be imported and transformations applied producing a modified GeoJSON
* Some support for reading native WSI formats: currently reads .czi and .scn but could be expanded to other formats supported by python package `tifffile`

