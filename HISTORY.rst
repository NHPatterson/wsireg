=======
History
=======

0.3.4 (2022-04-25)
---------------------
Major refactor of RegImage class, various small improvements throughout to support napari-wsireg.


0.3.2.2 (2022-02-04)
---------------------
This minor update adds the ability to draw binary or label masks using RegShapes.


0.3.2.1 (2021-12-28)
---------------------
This releases fixes some small bugs around shape transforms.

* fix geojson import/export
* fix bug with geojson reading


0.3.0 (2021-09-22)
-------------------

* add "ome.tiff-bytile" writer to write transformed images tile-by-tile
* unify data reading from tiffs to use `dask`
* numerous improvements, bug fixes, and additional tests
