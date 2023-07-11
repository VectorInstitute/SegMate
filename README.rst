.. start-in-sphinx-home-docs

=========================================
SegMate: A Segment Anything Model Toolkit
=========================================

This repository is a toolkit for using the Segment Anything Model (SAM) developed by Meta AI. It is capable of accurately "cutting out" any object from an image with just a single click.

SAM is a promptable segmentation system that exhibits zero-shot generalization to unfamiliar objects and images, eliminating the need for additional training. By providing prompts that specify what needs to be segmented in an image, SAM can perform a wide range of segmentation tasks without requiring additional training. 

Features
========

- **Easy inference** with SAM, supporting bounding boxes, points, masks, and text prompts
- **Automatic masking** without the need for prompts
- **Finetune** SAM on custom datasets
- **GroundingDINO integration** for text prompt segmentation

.. end-in-sphinx-home-docs

.. start-in-sphinx-sam-architecture

SAM Architecture
================

The design of SAM is highly flexible and enables seamless integration with other systems. It has been trained on millions of images and masks using a model-in-the-loop "data engine" approach. Researchers iteratively annotated images and updated the model, resulting in SAM's advanced capabilities and understanding of objects.

SAM was trained on over 1.1 billion segmentation masks collected from approximately 11 million licensed and privacy-preserving images. The dataset was built using SAM's ambiguity-aware design and a grid of points, allowing it to automatically annotate new images.

SAM utilizes a sophisticated architecture that enables efficient and accurate object segmentation. The model is designed with a three-step process involving an image encoder, a prompt encoder, and a mask decoder.

Image Encoder
-------------

The image encoder is responsible for capturing essential features from the input image. It processes the image through deep neural networks, extracting high-level representations that encode relevant information about objects and their context. This encoding step allows SAM to understand the visual content of the image.

Prompt Encoder
--------------

The prompt encoder plays a crucial role in SAM's promptable design. It processes the user-provided prompts, which can be bounding boxes, points, or text, and transforms them into meaningful representations. These representations serve as guidance for SAM to understand the desired object to be segmented. By encoding prompts, SAM gains the ability to perform a wide range of segmentation tasks without requiring additional training.

Mask Decoder
------------

The mask decoder takes the encoded information from both the image encoder and the prompt encoder to generate precise segmentation masks. It leverages a lightweight network that efficiently processes the encoded features and produces detailed object boundaries. The mask decoder's computational efficiency enables it to run seamlessly in a web browser, allowing for near real-time segmentation results.

.. end-in-sphinx-sam-architecture

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
    
    from segmate import SegMate
    from object_detector import GroundingDINO
    import utils

    od = GroundingDINO()
    sm = SegMate(model_type='MODEL_TYPE', checkpoint='PATH_to_MODEL', device='cuda', object_detector=od)


2. Perform segmentation with bounding box prompts:

.. code-block:: python

    masks = sm.segment(image=input_image, boxes_prompt=bbox)
    utils.show_masks(masks)


3. Perform segmentation with a text prompt:

.. code-block:: python

    masks = sm.segment(image=input_image, text_prompt=["building", 0.30, 0.25])
    utils.show_masks(masks)


4. Perform segmentation with point prompts:

.. code-block:: python

    masks = sm.segment(image=input_image, points_prompt=(point_coords, point_labels))
    utils.show_masks(masks)


5. Perform segmentation with a mask prompt:

.. code-block:: python

    masks = sm.segment(image=input_image, mask_prompt=mask)
    utils.show_masks(masks)


6. Generate masks automatically without prompts:

.. code-block:: python

    masks = sm.visualize_automask(image=input_image, mask_input=input_masks)
    utils.show_masks(masks)


7. Fine-tune the SAM model on a custom dataset:

.. code-block:: python

    sm.fine_tune(
        train_data=train_dataset, 
        original_input_size=500, 
        criterion=loss, 
        optimizer=optim, 
        lr=1e-5, 
        num_epochs=10)

.. end-in-sphinx-getting-started

Documentation
=============

Detailed package documentation: `SegMate Docs <https://segmate.readthedocs.io>`_

If you have any questions or need assistance, please don't hesitate to reach out to our support team or join our community forum. We hope you find this toolkit valuable and look forward to seeing the incredible applications you create with SAM!

License
=======
The code in this repository is published under 3-Clause BSD license (see ``LICENSE`` file).