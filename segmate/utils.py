"""
This file contains utility functions for the SegMate package.
"""
from PIL import Image

import cv2
import numpy as np
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
from matplotlib import patches
import torch
from torch.nn import functional as F
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download


def show_bounding_boxes(image: np.ndarray, bounding_boxes: np.ndarray) -> None:
    """
    Displays an image with bounding boxes overlaid on top of it.
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


def transform_image(image: Image) -> torch.Tensor:
    """
    Transforms an image using standard transformations for image-based models.

    Parameters:
    image (Image): The PIL Image to be transformed.

    Returns:
    torch.Tensor: The transformed image as a tensor.
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    return image_transformed


def load_model_hf(
        repo_id: str,
        filename: str,
        ckpt_config_filename: str,
        device: str = 'cpu'
    ) -> torch.nn.Module:
    """
    Loads a model from HuggingFace Model Hub.

    Parameters:
    repo_id (str): Repository ID on HuggingFace Model Hub.
    filename (str): Name of the model file in the repository.
    ckpt_config_filename (str): Name of the config file for the model in the repository.
    device (str): Device to load the model onto. Default is 'cpu'.

    Returns:
    torch.nn.Module: The loaded model.
    """
    # Ensure the repo ID and filenames are valid
    assert isinstance(repo_id, str) and repo_id, "Invalid repository ID"
    assert isinstance(filename, str) and filename, "Invalid model filename"
    assert isinstance(ckpt_config_filename,
                      str) and ckpt_config_filename, "Invalid config filename"

    # Download the config file and build the model from it
    cache_config_file = hf_hub_download(
        repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)

    # Download the model checkpoint and load it into the model
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()

    return model


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


def show_image(image: np.ndarray, color_map: str='binary_r', show_axis: bool=False) -> None:
    """
    Shows the image.

    Args:
        image (numpy.ndarray): The image to be shown.
        color_map (str): The color map to be used.
        show_axis (bool): Whether to show the axis or not.
    """
    # showing the image
    plt.imshow(image, cmap=color_map)
    if not show_axis:
        plt.axis('off')
    plt.show()


def show_masks(image: np.ndarray, masks: np.ndarray) -> None:
    """
    Shows the masks.

    Args:
        image (numpy.ndarray): The image to be shown.
        masks (numpy.ndarray): The masks to be shown.
    """
    image_pil = Image.fromarray(image)
    # Adjusted for single channel
    mask_overlay = np.zeros_like(image[..., 0], dtype=np.uint8)  

    for i, mask in enumerate(masks):
        mask = mask[0, :, :]
        # Assign a unique value for each mask
        mask_overlay += ((mask > 0) * (i + 1)).astype(np.uint8)  

    # Normalize mask_overlay to be in [0, 255]
    mask_overlay = (mask_overlay > 0) * 255  # Binary mask in [0, 255]

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

def convert_bboxes2center_points(bboxes: np.ndarray) -> np.ndarray:
    """
    Converts the bounding boxes to center points.

    Args:
        bboxes (np.ndarray): The bounding boxes of the image.

    Returns:
        center_points (np.ndarray): The center points of the bounding boxes.
    """
    # converting the bounding boxes to center points
    center_points = np.zeros((bboxes.shape[0], 2))
    center_points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    center_points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

    return center_points