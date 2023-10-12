===============
Getting Started
===============

.. include:: ../README.rst
   :start-after: start-in-sphinx-getting-started
   :end-before: end-in-sphinx-getting-started

3. Perform segmentation with a text prompt:

.. code-block:: python

    bbox, _, _ = od.detect(input_image, "building", 0.30, 0.25)
    masks = sm.segment(image=input_image, boxes_prompt=bbox)
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

    masks = utils.visualize_automask(image=input_image, mask_input=input_masks)
    utils.show_masks(masks)


7. Fine-tune the SAM model on a custom dataset:

.. code-block:: python

    sm.fine_tune(
        original_input_size=500, 
        criterion=loss, 
        optimizer=optim, 
        train_loader=train_loader,
        prompt_type="points_prompt",
        num_epochs=10)