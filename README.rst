.. start-in-sphinx-home-docs

=========================================
SegMate: A Segmentation Toolkit
=========================================

.. image:: https://img.shields.io/pypi/v/segmate.svg
        :target: https://pypi.org/project/segmate

.. image:: https://readthedocs.org/projects/segmate/badge/?version=latest
        :target: https://segmate.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/segmate.svg
        :target: https://opensource.org/licenses/BSD-3-Clause
        :alt: License

This repository is a toolkit for using the `Segment Anything Model (SAM) <https://segment-anything.com>`_ developed by Meta AI. It is capable of accurately "cutting out" any object from an image with just a single click.

SAM is a promptable segmentation system that exhibits zero-shot generalization to unfamiliar objects and images, eliminating the need for additional training. By providing prompts that specify what needs to be segmented in an image, SAM can perform a wide range of segmentation tasks without requiring additional training. 

Features
========

- **Easy inference** with SAM, supporting bounding boxes, points, masks, and text prompts
- **Automatic mask generation** 
- **Finetune** SAM on custom datasets
- `GroundingDINO <https://github.com/IDEA-Research/GroundingDINO/tree/main>`_ **integration** for text prompt segmentation
- Training a **custom decoder** to auto segment a specific type of object
- Training a **prompt embedding** to auto segment a specific type of object

.. end-in-sphinx-home-docs

.. start-in-sphinx-getting-started

Installation
============

First, install ``groundingdino`` from its repository, this is a dependency for ``segmate``:

**NOTE**: There is an issue with the setup script in the GroundingDINO repository causing it not able to install ``torch`` properly, please `manually install PyTorch <https://pytorch.org/get-started/locally/>`_ for now. For other issues, refer to the `installation guide <https://github.com/IDEA-Research/GroundingDINO/tree/main#hammer_and_wrench-install>`_: 

.. code-block:: console

    pip install -U git+https://github.com/IDEA-Research/GroundingDINO.git

Then, install ``segmate`` from `PyPI <https://pypi.org/project/segmate/>`_:

.. code-block:: console

    pip install segmate


Example Usage
=============

To use the provided code snippets, follow the steps below:

1. Import the required modules and initialize the necessary objects:

.. code-block:: python

    import torch
    
    from segmate.segmenter import SAM
    from segmate.object_detector import GroundingDINO
    import segmate.utils as utils

    # Model checkpoint path for GroundingDINO is optional. If no path provided, it will download from HuggingFace
    od = GroundingDINO(device='cuda', ckpt_path='PATH_TO_CHECKPOINT')
    sm = SAM(model_type='MODEL_TYPE', checkpoint='PATH_to_CHECKPOINT', device='cuda')


2. Perform segmentation with bounding box prompts:

.. code-block:: python

    masks = sm.segment(image=input_image, boxes_prompt=bbox)
    utils.show_masks(image, masks)

.. end-in-sphinx-getting-started

Documentation
=============

Detailed package documentation: `SegMate Docs <https://segmate.readthedocs.io>`_

If you have any questions or need assistance, please don't hesitate to reach out to our support team or join our community forum. We hope you find this toolkit valuable and look forward to seeing the incredible applications you create with SAM!

License
=======
The code in this repository is published under 3-Clause BSD license (see ``LICENSE`` file).
