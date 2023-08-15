"""
This file contains all object detector classes.
"""
from abc import ABC, abstractmethod
from PIL import Image

import torch
import numpy as np
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download

import segmate.utils as utils


class ObjectDetector(ABC):
    """
    An abstract class for all object detectors.
    """
    def __init__(self, device, ckpt_path):
        """
        Constructor for the ObjectDetector class.

        Args:
            device: The device to use.

        Returns:
            None
        """
        self.device = device
        self.model = self.load_model(ckpt_path)

    @abstractmethod
    def load_model(self, ckpt_path):
        """
        Load the model.
        """

    @abstractmethod
    def detect(self, image_np, text_prompt, box_threshold, text_threshold):
        """
        Run object detection.
        """


class GroundingDINO(ObjectDetector):
    """
    A class for the GroundingDINO object detector.
    """
    def __init__(
            self,
            device:str="cuda",
            ckpt_path:str=None
        ) -> None:
        """
        Constructor for the GroundingDINO class.

        Args:
            device: The device to use.

        Returns:
            None
        """
        super().__init__(device, ckpt_path)

    def load_model(self, ckpt_path=None) -> torch.nn.Module:
        """
        Build the GroundingDINO model.
        """
        # Download the config file and build the model from it
        ckpt_repo = "ShilongLiu/GroundingDINO"
        ckpt_config = "GroundingDINO_SwinB.cfg.py"

        cache_config_file = hf_hub_download(
            repo_id=ckpt_repo, filename=ckpt_config)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        model.to(self.device)

        if not ckpt_path:
            ckpt_path = hf_hub_download(repo_id=ckpt_repo, filename="groundingdino_swinb_cogcoor.pth")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
            
        return model

    def transform_image(self, image: Image) -> torch.Tensor:
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

    def detect(
            self,
            image_np: np.ndarray,
            text_prompt: str,
            box_threshold: float,
            text_threshold: float
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the GroundingDINO model prediction.

        Args:
            image_np: Input PIL Image.
            text_prompt: Text prompt for the model.
            box_threshold: Box threshold for the prediction.
            text_threshold: Text threshold for the prediction.

        Returns:
            Tuple containing boxes, logits, and phrases.
        """
        image_pil = Image.fromarray(image_np)
        image_trans = self.transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.model,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        width, height = image_pil.size
        # rescale boxes
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
            [width, height, width, height])

        return boxes, logits, phrases
