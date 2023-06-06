from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import utils
import numpy as np

class BISDataset(Dataset):
    """
    A PyTorch Dataset that provides access to the Building Image Segmentation (BIS) Dataset.
    """
    def __init__(self, dataset, preprocess, img_size, device):
        self.dataset = dataset
        self.preprocess = preprocess
        self.img_size = img_size
        self.device = device

    def __len__(self):
        return len(self.dataset)


    def convert_bboxes(self, bboxes):
        converted_bboxes = []
        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            converted_bboxes.append([x_min, y_min, x_max, y_max])
        return np.array(converted_bboxes)

    def __getitem__(self, idx):
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
        box_prompt = self.convert_bboxes(box_prompt)
        box_prompt = torch.as_tensor(box_prompt, device=self.device)
        
        # get the ground truth segmentation mask
        gt_mask = utils.get_segmentation_mask(item["objects"]).reshape((1, mask_size, mask_size)).astype('float32')
        gt_mask = torch.as_tensor(gt_mask, device=self.device)
    
        return input_image, box_prompt, gt_mask