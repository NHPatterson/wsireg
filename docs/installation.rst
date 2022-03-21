.. highlight:: shell

============
Installation
============

Installing wsireg should be painless. If you are completely new to python
see the end of this page for a brief how-to get started.

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

Complete newbie guide
----------------------


First let's install python and conda. Go to https://docs.conda.io/en/latest/miniconda.html and
find the correct installer for your system.

Now, let's set up a virtual environment. This makes sure all of our packages and their dependencies
are compatible with one another. Set up a virtual environment by opening your terminal (mac, linux) or opening
the anaconda prompt (windows).
Once in the terminal / command prompt, use the following lines to install the necessary packages.

.. code-block:: bash

    conda create -y -n wreg python=3.8
    conda activate wreg
    pip install wsireg


After this, we can call wsireg from the command line or use it as a library.

Do this before you take any action each time you want to use wsireg:

.. code-block:: bash

    # activate environment
    conda activate wreg
    # use wsireg
    wsireg2d "path/to/config/file.yaml"





