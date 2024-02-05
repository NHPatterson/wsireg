=================
Image Data Format
=================

2-D requirement
""""""""""""""""""
First and foremost, wsireg does not support any alignment of 3-D modalities, i.e., images with z-stacks. This
may change in the future.


All supported formats
"""""""""""""""""""""
wsireg supports 2D images that read in using the python tifffile or czi file library through dask and zarr. We started
to implement support using aicsimageio but found that default chunking was not always appropriate for tested whole slide images.
For now, the supported formats are:

- .ome.tiff - pyramidal OME-TIFF or regular OME-TIFF
- .czi - Zeiss images (Bright & Fluo)
- numpy/dask/zarr arrays - if you can process your image into an array-like container it can be used in wsireg
- .tiff - Generic tiff/big tiff files
- .scn - Leica SCN400
- .svs - Generic format developed by Aperio
- .ndpi - Hamamatsu slide scanner
- .jpeg/.png - These are read through ITK, not recommmended to use these as they fail to meet scientific imaging needs

Converting to OME-TIFF
"""""""""""""""""""""""""""""""""""""""

It is difficult to be comprehensive of all the different vendor and vendor-neutral image formats that exist
for whole slide images. wsireg supports a few vendor formats natively, but for others, the best path is to
convert them to OME-TIFF using the `bioformats2raw <https://github.com/glencoesoftware/bioformats2raw>`_,
`raw2ometiff <https://github.com/glencoesoftware/raw2ometiff>`_ pipeline developed by Glencoe Software which uses
bioformats' capability to read many vendor file formats and java parallelization to do the conversion rapidly.
This data pipeline will convert your data into a format that wsireg will understand without issue.

Note on image series
""""""""""""""""""""
A note on series: Many whole slide images have multiple image series within the data, these are separate images stored
alongside the image. Typically this is a macro image (low-res overview scan of the slide area) or label image (low-res scan
of just the label area). But some formats will also store each high-res scan area as its own series. wsireg does not support
multi-series images yet and will default to the largest series by XY plane (under the assumption this is the whole slide image).

If you know which series you'd like to register, you can optionally only convert it
using the `bioformats2raw <https://github.com/glencoesoftware/bioformats2raw>`_ pipeline like so:

.. code-block:: bash

    bioformats2raw --series 1 --resolutions 5 "my_image.czi" "my_image.h5"