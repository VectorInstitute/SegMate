"""
This file contains all object detector classes.
"""
from abc import ABC, abstractmethod
from PIL import Image

from groundingdino.util import box_ops
from groundingdino.util.inference import predict
import torch
import numpy as np

import utils


class ObjectDetector(ABC):
    """
    An abstract class for all object detectors.
    """
    def __init__(self, model_name, device):
        """
        Constructor for the ObjectDetector class.
        """
        self.model_name = model_name
        self.device = device
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """
        Load the model.
        """

    @abstractmethod
    def predict(self, image_np, text_prompt, box_threshold, text_threshold):
        """
        Run the model prediction.
        """


class GroundingDINO(ObjectDetector):
    """
    A class for the GroundingDINO object detector.
    """
    def __init__(
            self,
            model_name:str="groundingdino",
            device:str="cuda"
        ) -> None:
        """
        Constructor for the GroundingDINO class.
        """
        super().__init__(model_name, device)

    def load_model(self) -> torch.nn.Module:
        """
        Build the GroundingDINO model.
        """
        ckpt_repo = "ShilongLiu/GroundingDINO"
        ckpt_file = "groundingdino_swinb_cogcoor.pth"
        ckpt_config = "GroundingDINO_SwinB.cfg.py"
        return utils.load_model_hf(ckpt_repo, ckpt_file, ckpt_config, self.device)

    def predict(
            self,
            image_np: np.ndarray,
            text_prompt: str,
            box_threshold: float,
            text_threshold: float
        ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Run the GroundingDINO model prediction.

        Parameters:
        image_np (np.ndarray): Input PIL Image.
        text_prompt (str): Text prompt for the model.
        box_threshold (float): Box threshold for the prediction.
        text_threshold (float): Text threshold for the prediction.

        Returns:
        Tuple containing boxes, logits, and phrases.
        """
        image_pil = Image.fromarray(image_np)
        image_trans = utils.transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.model,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        width, height = image_pil.size
        # rescale boxes
        boxes = box_ops.box_cxcywh_to_xyxy(
            boxes) * torch.Tensor([width, height, width, height]).to(boxes.device)
        return boxes, logits, phrases
