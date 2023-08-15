"""
This file contains the code for all dataset classes.
"""
from typing import Callable

from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import numpy as np
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

        # get the ground truth segmentation mask
        gt_mask = utils.get_segmentation_mask(item["objects"]['segmentation'], mask_size)
        gt_mask = gt_mask.reshape((1, mask_size, mask_size)).astype('float32')
        gt_mask = torch.as_tensor(gt_mask, device=self.device)

        return input_image, box_prompt, gt_mask