"""
This file contains utility functions for the SegMate package.
"""
from PIL import Image
from typing import Union

import cv2
import numpy as np
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
from matplotlib import patches


def show_bounding_boxes(image: np.ndarray, bounding_boxes: np.ndarray) -> None:
    """
    Displays an image with bounding boxes overlaid on top of it.

    Args:
        image (numpy.ndarray): The image to display.
        bounding_boxes (numpy.ndarray): The bounding boxes to overlay on top of the image.

    Returns:
        None
    """
    # Create a figure and axes
    _, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Plot bounding boxes
    for bbox in bounding_boxes:
        x, y, width, height = bbox
        rect = patches.Rectangle(
            (x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Configure plot settings
    ax.axis('off')
    plt.show()


def get_segmentation_mask(segmentation_label: list, size: int) -> np.ndarray:
    """
    Converts a COCO segmentation label to a binary mask.

    Args:
        segmentation_label (list): A COCO segmentation label
        size (int): The size of the the mask

    Returns:
        numpy.ndarray: A binary mask, where the mask is represented as a 2D numpy array of shape
        (height, width).
    """
    # Convert COCO segmentation label to binary mask
    binary_mask = np.zeros((size, size, 1))
    for seg_lbl in segmentation_label:
        rle = coco_mask.frPyObjects(seg_lbl, size, size)
        binary_mask += coco_mask.decode(rle)
    binary_mask = binary_mask.reshape((1, size, size)).astype('float32')

    return binary_mask


def convert_coco_to_sam(bboxes: list) -> np.ndarray:
    """
    Converts a list of bounding boxes from [x_min, y_min, width, height] format to
    [x_min, y_min, x_max, y_max] format.
    (Coco format is [x_min, y_min, width, height] and SAM input format is
    [x_min, y_min, x_max, y_max])

    Args:
        bboxes (list): A list of bounding boxes, where each bounding box is represented as
        [x_min, y_min, width, height].

    Returns:
        list: A list of converted bounding boxes, where each bounding box is represented as
        [x_min, y_min, x_max, y_max].
    """
    converted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        converted_bboxes.append([x_min, y_min, x_max, y_max])
    return np.array(converted_bboxes)


def convert_sam_to_coco(bboxes: np.ndarray) -> np.ndarray:
    """
    Converts a list of bounding boxes from SAM input format to COCO format.

    Args:
        bboxes (np.ndarray): A list of bounding boxes, where each bounding box is represented as
        [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: A list of converted bounding boxes, where each bounding box is represented as
        [x_min, y_min, width, height].
    """
    converted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        converted_bboxes.append([x_min, y_min, width, height])
    return np.array(converted_bboxes)


def show_anns(anns: list, ax: plt.axes) -> None:
    """
    Displays a list of annotations on a matplotlib axis. Each annotation is displayed as a random
    color polygon.

    Args:
        anns (list): A list of annotations, where each annotation is represented as a dictionary
        with the following keys:
            - segmentation (list): A list of polygons, where each polygon is represented as a list
                of points.
            - area (float): The area of the annotation.
        ax (matplotlib.axes.Axes): The matplotlib axis to display the annotations on.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)

    img_shape = (sorted_anns[0]['segmentation'].shape[0],
                 sorted_anns[0]['segmentation'].shape[1])
    img = np.ones((img_shape[0], img_shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def load_image(image_path: str) -> np.ndarray:
    """
    Loads the image.

    Args:
        image_path (str): The path to the image.

    Returns:
        image (numpy.ndarray): The loaded image.
    """
    # loading the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def show_image(
    img: np.ndarray,
    size: int=8,
    show_axis: bool=False, 
    inter: int=500,
    points: tuple[list, list]=(None, None)
) -> None:
    """
    Shows the image and points if provided.

    Args:
        img (numpy.ndarray): The image to be shown.
        img_section (str): Show only a quarter of the image, options are 'top_left', 'top_right',
            'bottom_left', 'bottom_right'.
        size (int): Size of plot.
        show_axis (bool): Whether to show the axis or not.
        inter (int): Tick interval for axis.
        points (tuple): Tuple of point coordinates and labels to be plotted on the image,
            label '1' stands for foreground points, label '0' stands for background points.
    """
    height, width, _ = img.shape
        
    f = plt.figure(figsize=(size, size))
    plt.xticks(np.arange(0, width, inter))
    plt.yticks(np.arange(0, height, inter))
    
    point_coords, point_labels = points
    if point_coords is not None and point_labels is not None:
        for pt, lbl in zip(point_coords, point_labels):
            style = "bo" if lbl == 1 else "ro"
            plt.plot(pt[0], pt[1], style)
    
    plt.imshow(img)
    if not show_axis:
        plt.axis('off')
    plt.show()


def show_masks(
        image: np.ndarray,
        masks: np.ndarray, 
        additional_masks: list[np.ndarray]=None,
        size: int=None
    ) -> None:
    """
    Shows the masks.

    Args:
        image (numpy.ndarray): The image to be shown.
        masks (numpy.ndarray): The mask overlayed image to be shown.
        additional_masks (list(numpy.ndarray)): Additional masks to be shown, usually for comparison.
        size (int): The size of the plot.
    """

    all_masks = [masks]

    count = 2
    if additional_masks:
        count += len(additional_masks)
        all_masks.extend(additional_masks)

    if size:
        plt.figure(figsize=(size * count, size))

    image_pil = Image.fromarray(image)
    
    # Plot original image
    plt.subplot(1, count, 1)
    plt.imshow(image_pil)
    plt.axis('off')

    # Plot masks    
    for i, mask_set in enumerate(all_masks):
        mask_overlay = np.zeros_like(image[..., 0], dtype=np.uint8)

        for j, mask in enumerate(mask_set):
            mask = mask[0, :, :]
            # Assign a unique value for each mask
            mask_overlay += ((mask > 0) * (j + 1)).astype(np.uint8)

        # Normalize mask_overlay to be in [0, 255]
        mask_overlay = (mask_overlay > 0) * 255  # Binary mask in [0, 255]

        plt.subplot(1, count, i + 2)
        plt.imshow(image_pil)
        plt.imshow(mask_overlay, cmap='viridis', alpha=0.4)  # Overlay the mask with some transparency
        plt.axis('off')
    
    plt.show()


def show_points(image: np.ndarray, point_coords: list, point_labels: list) -> None:
    """
    Shows the points.

    Args:
        image (numpy.ndarray): The image to be shown.
        point_coords (list): The coordinates of the points.
        point_labels (list): The labels of the points.
    """
    for pt, lbl in zip(point_coords, point_labels):
        style = "bo" if lbl == 1 else "ro"
        plt.plot(pt[0], pt[1], style)

    plt.imshow(Image.fromarray(image))


def binarize_mask(
    masks: np.ndarray,
    sum_all_masks: bool = True
) -> np.ndarray:
    """
    Post-processes the segmentation mask to the original input size.

    Args:
        masks (np.ndarray): The generated segmentation masks.
        sum_all_masks (bool): Whether to sum all the masks or not.

    Returns:
        binary_mask (np.ndarray): The binarized segmentation mask of the image.
    """
    # post-processing the segmentation mask
    masks = masks.sum(axis=1)
    thresholded_mask = np.where(masks > 0.0, 1, 0)
    binary_mask = thresholded_mask.astype(np.int8)

    if sum_all_masks:
        binary_mask = np.sum(binary_mask, axis=0)
        binary_mask = np.where(binary_mask > 1, 1, binary_mask)

    return binary_mask


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """
    Saves the segmentation mask to the specified output path.

    Args:
        binary_mask: The binarized segmentation mask of the image.
        output_path: The path to save the segmentation map.

    Returns:
        None
    """
    # saving the segmentation mask
    cv2.imwrite(output_path, mask)


def visualize_automask(
    image: Union[str, np.ndarray],
    masks: np.ndarray,
    output_path: str = None
) -> None:
    """
    Visualizes the segmentation mask on the image and saves the resulting visualization.

    Args:
        image: The image or the path to the image to be segmented.
        mask: The segmentation mask to be visualized.
        output_path: The path to save the resulting visualization.

    Returns:
        None
    """
    # loading the image if the input is a path to an image
    if isinstance(image, str):
        image = load_image(image)

    # Create a new figure with two axes
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display the first image on the first axis
    plt.axis('off')
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Display the second image on the second axis
    plt.axis('off')
    ax2.imshow(image)
    show_anns(masks, ax2)
    ax2.set_title('Image with Masks')
    ax2.axis('off')

    if output_path is not None:
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight')


def get_masks_size(masks: np.ndarray) -> float:
    """
    Calculates the relative size of the masks.

    Args:
        masks (np.ndarray): The generated segmentation masks.

    Returns:
        mask_relative_size (float): The relative size of the mask.
    """
    mask_overlay = np.zeros_like(masks[0, 0, ...], dtype=np.uint8)
    for mask in masks:
        mask = mask[0, :, :]
        # Stack the masks
        mask_overlay += (mask > 0).astype(np.uint8)

    mask_relative_size = np.count_nonzero(mask_overlay) / masks[0, 0, ...].size * 10000
    return mask_relative_size


def plot_mask_diff(masks_1: np.ndarray, masks_2: np.ndarray, size: int=None) -> None:
    """
    Plots the difference between two masks.

    Args:
        masks_1 (np.ndarray): The first mask.
        masks_2 (np.ndarray): The second mask.
        size (int): The size of the plot.

    Returns:
        None
    """
    if masks_1[0, ...].shape != masks_2[0, ...].shape:
        print("Mask size mismatch!")
        return
    mask_overlay_1 = np.zeros_like(masks_1[0, 0, ...], dtype=np.uint8)
    mask_overlay_2 = np.zeros_like(masks_2[0, 0, ...], dtype=np.uint8)
    
    for mask in masks_1:
        mask = mask[0, :, :]
        mask_overlay_1 += (mask > 0).astype(np.uint8)
    mask_overlay_1 = (mask_overlay_1 > 0) * 1
        
    for mask in masks_2:
        mask = mask[0, :, :]
        mask_overlay_2 += (mask > 0).astype(np.uint8)
    mask_overlay_2 = (mask_overlay_2 > 0) * 2
    
    mask_diff_overlay = (mask_overlay_1 + mask_overlay_2) * 127
    
    if size:
        plt.figure(figsize=(size, size))
    plt.imshow(mask_diff_overlay, cmap='viridis')  # Overlay the mask with some transparency
    plt.axis('off')
    plt.show()


def find_center_on_mask(mask: np.ndarray) -> tuple[int, int]:
    """
    Find the center point of a binary mask. If the center does not lie on the mask,
    find the nearest mask point to the center.

    Args:
    - mask (numpy.ndarray): A 2D binary mask.

    Returns:
    - tuple: (y, x) coordinates of the center point on the mask.
    """
    # Find where the mask is true
    rows, cols = np.where(mask)

    # Calculate bounding box
    top_row = np.min(rows)
    bottom_row = np.max(rows)
    left_col = np.min(cols)
    right_col = np.max(cols)

    # Calculate center of bounding box
    center_row = (top_row + bottom_row) // 2
    center_col = (left_col + right_col) // 2

    # Check if the center is on the mask
    if mask[center_row, center_col]:
        return center_row, center_col

    # If the center is not on the mask, find the nearest mask point
    distances = (rows - center_row)**2 + (cols - center_col)**2
    nearest_index = np.argmin(distances)

    return rows[nearest_index], cols[nearest_index]