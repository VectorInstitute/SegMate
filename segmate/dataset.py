"""
This file contains the code for all dataset classes.
"""
from typing import Callable

from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import numpy as np
import os
import cv2
from datasets import Dataset as HFDataset

import segmate.utils as utils


class BISDataset(Dataset):
    """
    A PyTorch Dataset that provides access to the Building Image Segmentation (BIS) Dataset.
    """

    def __init__(
        self,
        dataset: HFDataset,
        preprocess: Callable,
        img_size: int,
        device: str,
    ) -> None:
        """
        Constructor for the BISDataset class.
        
        Args:
            dataset: The HuggingFace dataset to use.
            preprocess: The preprocessing function to use.
            img_size: The size of the image to use.
            device: The device to use.

        Returns:
            None
        """
        self.dataset = dataset
        self.preprocess = preprocess
        self.img_size = img_size
        self.device = device

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the item at the given index.
        
        Args:
            idx: The index of the item to return.
        
        Returns:
            The item at the given index.
        """
        item = self.dataset[idx]
        image = np.array(item["image"])
        mask_size = image.shape[0]

        # prepare the image for the model
        transform = ResizeLongestSide(self.img_size)
        input_image = transform.apply_image(image)
        input_image = torch.as_tensor(input_image, device=self.device)
        input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

        # preprocess the image
        input_image = self.preprocess(input_image).squeeze()

        # prepare the prompt for the model
        box_prompt = np.array(item['objects']['bbox']).astype('float32')
        box_prompt = utils.convert_coco_to_sam(box_prompt)
        box_prompt = transform.apply_boxes(box_prompt, image.shape[:2])
        box_prompt = torch.as_tensor(box_prompt, device=self.device)
        box_prompt = box_prompt.squeeze()

        # get the ground truth segmentation mask
        gt_mask = utils.get_segmentation_mask(item["objects"]['segmentation'], mask_size)
        gt_mask = gt_mask.reshape((1, mask_size, mask_size)).astype('float32')
        gt_mask = torch.as_tensor(gt_mask, device=self.device)

        return input_image, box_prompt, gt_mask
    

    from typing import Callable


class AerialImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        preprocess: Callable,
        img_size: int,
        device: str,
    ) -> None:
        """
        Constructor for the Aerial Image Dataset class.
        
        Args:
            data_dir: The path to the dataset.
            preprocess: The preprocessing function to use.
            img_size: The size of the image to use.
            device: The device to use.

        Returns:
            None
        """
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.img_size = img_size
        self.device = device

        self.image_folder = os.path.join(self.data_dir, "images")
        self.mask_folder = os.path.join(self.data_dir, "gt")

        self.filenames = os.listdir(self.image_folder)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.filenames)

    def calculate_centers(self, contours):
        """
        Function to calculate center of a building mask
        """
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        return np.array(centers)

    def calculate_bounding_boxes(self, contours):
        """
        Function to calculate bounding boxes around all buildings
        """
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x+w, y+h))
        return np.array(bounding_boxes)


    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        
        Args:
            idx: The index of the item to return.
        
        Returns:
            The item at the given index.
        """
        img_name = os.path.join(self.image_folder, self.filenames[idx])
        mask_name = os.path.join(self.mask_folder, self.filenames[idx])

        image = cv2.imread(img_name)
        gt_mask = cv2.imread(mask_name, 0)
        
        # Threshold the image to make sure it's binary
        _, binary_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours which corresponds to the buildings
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate centers and bounding boxes
        centers = self.calculate_centers(contours)
        bounding_boxes = self.calculate_bounding_boxes(contours) # the bboxes are already in SAM format

        # prepare the image for the model
        transform = ResizeLongestSide(image.shape[0])
        input_image = transform.apply_image(image)
        input_image = torch.as_tensor(input_image, device=self.device)
        input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

        # preprocess the image
        input_image = self.preprocess(input_image).squeeze()
        
        # prepare the prompt for the model
        box_prompt = transform.apply_boxes(bounding_boxes, image.shape[:2])
        box_prompt = torch.as_tensor(box_prompt, device=self.device)
        box_prompt = box_prompt.squeeze()

        # get the ground truth segmentation mask
        gt_mask = gt_mask.reshape((1, gt_mask.shape[0], gt_mask.shape[0])).astype('float32')
        gt_mask = torch.as_tensor(gt_mask, device=self.device)

        return input_image, box_prompt, gt_mask