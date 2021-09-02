=======
History
=======

0.0.2 (2021)
------------------

* First release on PyPI.

0.2.1 (2021-04-14)
------------------

* add `RegImage` sub-classes for different file types
	- `TiffFileRegImage` (.scn, .ndpi,.tiff,.tif) : uses `dask` + `zarr` to do memory-efficient computation of necessary data for registration / transformation
	- `CziRegImage` (.czi) : Carl Zeiss image format, can perform read-time pre-processing like RGB -> greyscale or selection of individual channels to limit memory footprint
	- `OmeTiffRegImage` (.ome.tiff,ome.tif): uses TiffFile to read images and parses OME metadata to get interleaved RGB information
	- `MergeRegImage` (meta): used to transform multiple images' channels to a single OME-TIFF after registration if they output to the same size and data type (i.e. for cyclic IF)
	- `NpRegImage` (`np.ndarray`): Supports adding a registration image from a `numpy` array
	- `SitkRegImage` (everything else): uses SimpleITK to read images as a last resort. Will read entire image into memory!

* support masks for registration
	- Masks can be used in `elastix` to define pixels used in metric calculation
	- add ability to automatically crop images based on associated masks's bounding box (can be useful if image dimensions differ greatly)

* use `RegTransform` class to manage transformations

0.3.0 (2021-09-02)
------------------

* add "ome.tiff-bytile" writer to write transformed images tile-by-tile
* unify data reading from tiffs to use `dask`
