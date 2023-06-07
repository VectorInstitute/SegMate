from PIL import Image

import numpy as np
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download


def show_bounding_boxes(image, bounding_boxes):
    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Plot bounding boxes
    for bbox in bounding_boxes:
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Configure plot settings
    ax.axis('off')
    plt.show()


def get_segmentation_mask(segmentation_label, size):
    """
    Converts a COCO segmentation label to a binary mask.
    
    Args:
        segmentation_label (list): A COCO segmentation label
        size (int): The size of the the mask

    Returns:
        numpy.ndarray: A binary mask, where the mask is represented as a 2D numpy array of shape (height, width).
    """
    # Convert COCO segmentation label to binary mask
    binary_mask= np.zeros((size, size, 1))
    for seg_lbl in segmentation_label['segmentation']:
        rle = coco_mask.frPyObjects(seg_lbl, size, size)
        binary_mask += coco_mask.decode(rle)
    
    return binary_mask


def convert_bboxes(bboxes):
    """
    Converts a list of bounding boxes from [x_min, y_min, width, height] format to [x_min, y_min, x_max, y_max] format.
    (Coco format is [x_min, y_min, width, height] and SAM input format is [x_min, y_min, x_max, y_max])

    Args:
        bboxes (list): A list of bounding boxes, where each bounding box is represented as [x_min, y_min, width, height].

    Returns:
        list: A list of converted bounding boxes, where each bounding box is represented as [x_min, y_min, x_max, y_max].
    """
    converted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        converted_bboxes.append([x_min, y_min, x_max, y_max])
    return np.array(converted_bboxes)


def show_anns(anns, ax):
    """
    Displays a list of annotations on a matplotlib axis. Each annotation is displayed as a random color polygon.
    
    Args:
        anns (list): A list of annotations, where each annotation is represented as a dictionary with the following keys:
            - segmentation (list): A list of polygons, where each polygon is represented as a list of points.
            - area (float): The area of the annotation.
        ax (matplotlib.axes.Axes): The matplotlib axis to display the annotations on.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)

    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])
    img = np.ones((img_shape[0], img_shape[1], 4))
    img[:,:,3] = 0
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


def load_model_hf(repo_id: str, filename: str, ckpt_config_filename: str, device: str = 'cpu') -> torch.nn.Module:
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
    assert isinstance(ckpt_config_filename, str) and ckpt_config_filename, "Invalid config filename"
    
    # Download the config file and build the model from it
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    model.to(device)
    
    # Download the model checkpoint and load it into the model
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    
    return model
