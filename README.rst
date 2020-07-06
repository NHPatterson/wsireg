======
wsireg
======


.. image:: https://readthedocs.org/projects/wsireg/badge/?version=latest
        :target: https://wsireg.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


python package for registering multimodal whole slide microscopy images using SimpleElastix. This package helps to
make management of very complex registration tasks easier.

* Documentation: https://wsireg.readthedocs.io.


Features
--------

* Graph based approach to defining modalities and arbitrary transformation paths between associated images.
* Linear and non-linear transformation models
* Use of elastix (through SimpleElastix) to perform registration
* Transform associated data (masks, shape data) along the same path as the images

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
