.. highlight:: shell

============
Installation
============


Stable release
--------------

To install wsireg, run this command in your terminal:

.. code-block:: console

    $ pip install wsireg

This is the preferred method to install wsireg, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/



From sources
------------

The sources for wsireg can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/nhpatterson/wsireg

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/nhpatterson/wsireg/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/nhpatterson/wsireg
.. _tarball: https://github.com/nhpatterson/wsireg/tarball/master

SimpleElastix dependency
------------------------

To register images using wsireg, you will need to install SimpleElastix, which, right now, requires compilation from
source. Please see `SimpleElastix documentation <https://simpleelastix.readthedocs.io/GettingStarted.html>`_. Most of the issues
for this project revolve around this compilation step and failure of users to do so.
Thus, I have compiled 64-bit Python 3.8 SimpleElastix wheels (that may work on previous versions of Python 3)
for Windows and Linux (sorry no macOS yet) for a very recent version, but I *CANNOT* guarantee they will work
on your system.
You can download these wheels (`Linux wheel <https://drive.google.com/file/d/1GgoaX5tBV8x4sydtWj-0MSCti1qp8LuV/view?usp=sharing>`_,
`Windows wheel <https://drive.google.com/file/d/1SBrVTMZ5XExQpi3Gs-3RcZNnwJgYv5gR/view?usp=sharing>`_)
and install them using from your terminal like so:

.. code-block:: console

    $ pip install /path/to/downloaded/simpleelastix.whl


