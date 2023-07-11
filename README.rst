.. start-in-sphinx-home-docs

=========================================
SegMate: A Segment Anything Model Toolkit
=========================================

This repository is a toolkit for using the `Segment Anything Model (SAM) <https://segment-anything.com>`_ developed by Meta AI. It is capable of accurately "cutting out" any object from an image with just a single click.

SAM is a promptable segmentation system that exhibits zero-shot generalization to unfamiliar objects and images, eliminating the need for additional training. By providing prompts that specify what needs to be segmented in an image, SAM can perform a wide range of segmentation tasks without requiring additional training. 

Features
========

- **Easy inference** with SAM, supporting bounding boxes, points, masks, and text prompts
- **Automatic masking** without the need for prompts
- **Finetune** SAM on custom datasets
- `GroundingDINO <https://github.com/IDEA-Research/GroundingDINO/tree/main>`_ **integration** for text prompt segmentation

.. end-in-sphinx-home-docs

.. start-in-sphinx-getting-started

Installation
============

To install ``segmate`` from `PyPI <https://pypi.org/project/segmate/>`_:

.. code-block:: console

    pip install segmate


Example Usage
=============

To use the provided code snippets, follow the steps below:

1. Import the required modules and initialize the necessary objects:

.. code-block:: python

    import torch
    
    from segmate.segmate import SegMate
    from segmate.object_detector import GroundingDINO
    import segmate.utils as utils

    od = GroundingDINO()
    sm = SegMate(model_type='MODEL_TYPE', checkpoint='PATH_to_MODEL', device='cuda', object_detector=od)


2. Perform segmentation with bounding box prompts:

.. code-block:: python

    masks = sm.segment(image=input_image, boxes_prompt=bbox)
    utils.show_masks(masks)

.. end-in-sphinx-getting-started

Documentation
=============

Detailed package documentation: `SegMate Docs <https://segmate.readthedocs.io>`_

If you have any questions or need assistance, please don't hesitate to reach out to our support team or join our community forum. We hope you find this toolkit valuable and look forward to seeing the incredible applications you create with SAM!

License
=======
The code in this repository is published under 3-Clause BSD license (see ``LICENSE`` file).
