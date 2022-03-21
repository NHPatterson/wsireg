==================
Command Line Usage
==================

.. role:: ilyaml(code)
   :language: yaml

In addition to the use of wsireg as a library that can be scripted within the python intepreter,
wsireg can be fully configured and run using the command line in supplied a YAML configuration file. YAML
follows a similar structure to python where spaces and indents below keys (text followed by :ilyaml:`:`)
are nested keys and is easily human editable.

After you have completed the YAML configuration you can run registration as shown below. The :code:`wsireg2d` script
is installed as a console script in the python environment.

.. code-block:: bash

    wsireg2d "path/to/my-reg-file.yaml"


Definition of keys in wsireg configuration YAML
###############################################

An example YAML configuration file with comments explaining each key is below the explanatory text.

There are three top "project" level keys in the YAML configuration file.

#. :ilyaml:`project_name:` A short text string defining the project name that is prepended to all output files. (REQUIRED)

#. :ilyaml:`output_dir:` A text string to a directory where outputs will be saved. (REQUIRED)

#. :ilyaml:`cache_images:` Whether or not to save intermediate pre-processed registration images. (OPTIONAL)

Beyond project definition, the first required key is the :ilyaml:`modalities:` top-level key which starts the definition
of the registration images.Below the :ilyaml:`modalities` key and indented or spaced are the
definition of the modality names. The name is given in the key itself as shown in the snippet below.

.. code-block:: yaml

    modalities:
        fluo_image:
            image_filepath: fluo_im.tiff
            image_res: 0.325
        fluo_image2:
            image_filepath: fluo_im2.tiff
            image_res: 0.325


Please see example below for cues on different settings within the modality for preprocessing.

Second, there is the :ilyaml:`reg_paths` key. Following this key, we define, with uniquely named sub-keys that give direction
to the registrations in the graphs, i.e. register image 1 to image 2, register image 3 to image 1 but through image 2, etc.
In the full example at the end of this document :ilyaml:`reg_path_0` and :ilyaml:`reg_path_1` are used to define
two registration paths, but these keys only require a unique name.

.. code-block:: yaml

    reg_paths:
      reg_path_0:
        src_modality_name: fluo_image
        tgt_modality_name: fluo_image2
        thru_modality: null
        reg_params:
            - rigid
            - affine
      reg_path_1:
        src_modality_name: fluo_image3
        thru_modality: fluo_image
        tgt_modality_name: fluo_image2
        reg_params:
            - rigid
            - affine


Complete YAML example
#####################

This is a complete example from a real project aligning three WSIs. It incorporates registration
paths where im2 -> im1 and im3 -> im2 -> im1.

.. code-block:: yaml

    # determines the file names for your project
    # i.e. apl-demo-postAF-registered.ome.tiff
    project_name: apl-demo
    # where you want outputs to go
    output_dir: D:/temp
    # whether to save preprocessed data currently on disk
    cache_images: true
    # top level for all images to be included in registration
    modalities:
      # top level key is the NAME that will be used in output files
      postAF:
        # absolute path to the image
        image_filepath: C:/Users/pattenh1/Dropbox (VU Basic Sciences)/APOLLO/demo/apl-test-wpad.ome.tiff
        # image resolution (i.e., pixel spacing):
        image_res: 0.65
        # list of names, must match length of channels
        # otherwise will become C1, C2, etc.
        # no channel names for RGB images
        channel_names:
            - postIMS eGFP
            - postIMS Brightfield
        # preprocessing settings
        preprocessing:
          # fluorescence image type (options: FL, BF)
          image_type: FL
          # enhance contrast automatically (options: true, false)
          contrast_enhance: true
          # change data type to consume less memory (options: true, false, recommended!)
          as_uint8: true
          # which channels to load in for reg
          ch_indices: 0 # (options: null, single integer, or integer list, if null, max intensity project is performed)
          # multiple channels:
          #ch_indices:
          # - 0
          # - 1

      preAF:
        image_filepath: C:/Users/pattenh1/Dropbox (VU Basic Sciences)/APOLLO/demo/microscopy/P1-D3-treatment_lipids.czi
        image_res: 0.65
        channel_names:
            - DAPI autofluorescence
            - eGFP autofluorescence
            - DsRed autofluorescence
        preprocessing:
          image_type: FL
          contrast_enhance: true
          as_uint8: true
          ch_indices: 1

      PAS:
        image_filepath: C:/Users/pattenh1/Dropbox (VU Basic Sciences)/APOLLO/demo/microscopy/LT3_2021-08-17 16_25_03.scn
        image_res: 0.5
        preprocessing:
          image_type: BF
          as_uint8: true
          # rotate the image X degrees counter clockwise
          rot_cc: -90 # (options: any number, if negative, rotates clockwise)
          # flip image example
          # flip: h # (options: h,v for horizontal or vertical coordinate flips)

    # top level key for what registrations should be performed
    reg_paths:
      # first alignment
      reg_path_0:
        # image that will be transformed
        src_modality_name: preAF
        # image that wwill be aligned to
        tgt_modality_name: postAF
        # whether image should be piped through another modality
        thru_modality: null
        # what transformation models to use
        # options: rigid, affine, similarity, nl, nl_mid
        # nl is non-rigid transformations recommended for serial sections!
        reg_params:
            - rigid
            - affine
      reg_path_1:
        src_modality_name: PAS
        thru_modality: preAF
        tgt_modality_name: postAF
        reg_params:
            - rigid
            - affine


