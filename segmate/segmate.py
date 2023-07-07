"""
This file contains the SegMate class
"""
from statistics import mean
from typing import Union

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2
import numpy as np

import utils
from object_detector import ObjectDetector


class SegMate:
    """
    Class for interacting with the Segment Anything Model (SAM) for image segmentation.
    """

    def __init__(
        self,
        model_type: str = 'vit_b',
        checkpoint: str = 'sam_vit_b_01ec64.pth',
        device: str = 'cuda',
        object_detector: ObjectDetector = None,
    ) -> None:
        """
        Initializes the SamKit object with the provided model path.

        :param model_type: The type of SAM: (vit_b, vit_l, vit_h) -> b: base, l: large, h: huge.
        :param checkpoint: The path to the pre-trained SAM.
        :param device:  The device to load the model on (default: cuda).

        :return: None
        """
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.object_detector = object_detector

        self.sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint)
        self.sam.to(self.device)

        self.predictor = SamPredictor(self.sam)

    def set_object_detector(self, object_detector: ObjectDetector) -> None:
        """
        Sets the object detector to the SegMate instance.
        """
        self.object_detector = object_detector

    def save_mask(self, binary_mask: np.ndarray, output_path: str) -> None:
        """
        Saves the resulting segmentation mask to the specified output path.

        :param binary_mask: The binarized segmentation mask of the image.
        :param output_path: The path to save the segmentation map.

        :return: None
        """
        # saving the segmentation mask
        cv2.imwrite(output_path, binary_mask)

    def preprocess_input(
        self,
        image: np.ndarray,
        bbox_prompt: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transform and preprocesses the input image and bounding boxes to the required size and
        converts them to tensors.

        :param image: The image to be transformed.
        :param bbox_prompt: The bounding boxes to be transformed.

        :return image: The transformed image.
        :return bbox_prompt: The transformed bounding boxes.
        """
        # transforming the image to the required size and preprocess it
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image = torch.as_tensor(input_image, device=self.device)
        input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.sam.preprocess(input_image)

        # transforming the bounding boxes to the required size
        bbox_prompt = transform.apply_boxes(bbox_prompt, image.shape[:2])
        bbox_prompt = torch.as_tensor(
            bbox_prompt, dtype=torch.float, device=self.device)

        return input_image, bbox_prompt

    def encode_input(
        self,
        input_image: torch.Tensor,
        bbox_prompt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes the input image and bounding boxes.

        :param image: The transformed image.
        :param bbox_prompt: The transformed bounding boxes.

        :return image_embedding: The encoded image.
        :return sparse_embeddings: The encoded sparse bounding boxes.
        :return dense_embeddings: The encoded dense bounding boxes.
        """
        # encoding the image with sam's image encoder
        image_embedding = self.sam.image_encoder(input_image)

        # encoding the prompt with sam's prompt encoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=bbox_prompt.squeeze(0),
            masks=None,
        )

        return image_embedding, sparse_embeddings, dense_embeddings

    def postprocess_mask(
        self,
        low_res_masks: torch.Tensor,
        transformed_input_size: tuple[int, int],
        original_input_size: tuple[int, int]
    ) -> np.ndarray:
        """
        Post-processes the segmentation mask to the original input size.

        :param low_res_masks: The generated segmentation mask.
        :param transformed_input_size: The size of the transformed input image.
        :param original_input_size: The size of the original input image.

        :return binary_mask: The binarized segmentation mask of the image.
        """
        # post-processing the segmentation mask
        upscaled_masks = self.sam.postprocess_masks(
            low_res_masks, transformed_input_size, original_input_size)
        thresholded_mask = F.threshold(upscaled_masks, 0.0, 0)
        binary_mask = F.normalize(thresholded_mask)
        binary_mask = binary_mask.sum(axis=0).unsqueeze(0)

        return binary_mask

    def generate_mask(
        self,
        image_embedding: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
        multimask_output: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the segmentation mask.

        :param image_embedding: The encoded image.
        :param sparse_embeddings: The encoded sparse bounding boxes.
        :param dense_embeddings: The encoded dense bounding boxes.

        :return low_res_masks: The generated segmentation mask.
        :return iou_predictions: The generated IOU predictions.
        """
        # generating the segmentation mask from the image and the prompt embeddings
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions

    def segment(
        self,
        image: Union[str, np.ndarray],
        text_prompt: tuple[str, float, float] = None,
        boxes_prompt: np.ndarray = None,
        points_prompt: tuple[np.ndarray, np.ndarray] = (None, None),
        mask_input: np.ndarray = None,
    ) -> np.ndarray:
        """
        Performs image segmentation using the loaded image and input prompt.

        :param image: The image or the path to the image to be segmented.
        :param text_prompt: The text prompt to be used for segmentation. The
                tuple contains the text prompt, the box threshold, and the text threshold.
        :param boxes_prompt: The bounding boxes prompt to be used for segmentation.
        :param points_prompt: The points prompt to be used for
                segmentation. The tuple contains the point coordinates and the point labels.
        :param mask_input: The mask input to be used for segmentation.

        :return binary_mask: The binarized segmentation mask of the image.
        """
        # setting the model to evaluation mode
        self.sam.eval()

        # loading the image if the input is a path to an image
        if isinstance(image, str):
            image = utils.load_image(image)

        self.predictor.set_image(image)

        # converting the text prompt to a list of bounding boxes with a zero-shot object detection
        # model (Grounding Dino, etc.)
        if text_prompt is not None:
            text, box_threshold, text_threshold = text_prompt
            boxes_prompt, _, _ = self.object_detector.predict(
                image, text, box_threshold, text_threshold)
        if boxes_prompt is not None:
            boxes_prompt = torch.tensor(boxes_prompt).to(self.device)
            boxes_prompt = self.predictor.transform.apply_boxes_torch(
                boxes_prompt, image.shape[:2])
        point_coords, point_labels = points_prompt
        if point_coords is not None and point_labels is not None:
            # point_coords, point_labels = points_prompt
            point_coords = torch.tensor(point_coords).to(
                self.device).unsqueeze(1)
            point_labels = torch.tensor(point_labels).to(
                self.device).unsqueeze(1)
            point_coords = self.predictor.transform.apply_coords_torch(
                point_coords, image.shape[:2])
        if mask_input is not None:
            mask_input = torch.tensor(mask_input).to(self.device)
            mask_input = mask_input[None, :, :, :]

        # performing image segmentation with sam
        masks, _, _ = self.predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes_prompt,
            mask_input=mask_input,
        )
        masks = masks.detach().cpu().numpy().astype(np.uint8)

        return masks

    def auto_segment(
        self,
        image: Union[str, np.ndarray],
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 100
    ) -> np.ndarray:
        """
        Performs image segmentation using the automatic mask generation method.

        :param image: The image or the path to the image to be segmented.
        :param points_per_side: The number of points per side of the mask.
        :param pred_iou_thresh: The IOU threshold for the predicted mask.
        :param stability_score_thresh: The stability score threshold for the predicted mask.
        :param crop_n_layers: The number of layers to crop from the image.
        :param crop_n_points_downscale_factor: The downscale factor for the number of points to
                crop from the image.
        :param min_mask_region_area: The minimum area of the mask region.

        :return masks: The generated segmentation mask of the image.
        """
        # creating the automatic mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

        # loading the image if the input is a path to an image
        if isinstance(image, str):
            image = utils.load_image(image)

        masks = mask_generator.generate(image)

        return masks

    def visualize_automask(
        self,
        image: Union[str, np.ndarray],
        masks: np.ndarray,
        output_path: str = None
    ) -> None:
        """
        Visualizes the segmentation mask on the image and saves the resulting visualization.

        :param image: The image or the path to the image to be segmented.
        :param mask: The segmentation mask to be visualized.
        :param output_path: The path to save the resulting visualization.

        ;return: None
        """
        # loading the image if the input is a path to an image
        if isinstance(image, str):
            image = utils.load_image(image)

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
        utils.show_anns(masks, ax2)
        ax2.set_title('Image with Masks')
        ax2.axis('off')

        if output_path is not None:
            # Save the figure
            plt.savefig(output_path, bbox_inches='tight')

    def get_dataset(self, dataset: Dataset) -> torch.utils.data.DataLoader:
        """
        Prepare the data of the desired set.

        :param dataset: The dataset to be prepared.

        :return: The data loader of the desired set.
        """

        # creating the data loader
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)

    def forward_pass(self, input_image: np.ndarray, bbox_prompt: np.ndarray, original_input_size: int) -> np.ndarray:
        """
        Performs a forward pass on the image and the prompt.

        :param image: The image to be segmented.
        :param bbox_prompt: The bounding boxes prompt to be used for segmentation.

        :return binary_mask: The binarized segmentation mask of the image.
        """
        # preprocessing and transforming the image and bounding box prompt(s) with sam's functions
        # input_image, bbox_prompt = self.preprocess_input(image, bbox_prompt)

        # encoding the image and the prompt with sam's encoders
        image_embedding, sparse_embeddings, dense_embeddings = self.encode_input(
            input_image, bbox_prompt)

        # generating the segmentation mask from the image and the prompt embeddings
        low_res_masks, _ = self.generate_mask(
            image_embedding, sparse_embeddings, dense_embeddings)

        # postprocessing the segmentation mask and converting it to a numpy array
        binary_mask = self.postprocess_mask(low_res_masks, transformed_input_size=tuple(
            input_image.shape[-2:]), original_input_size=original_input_size)

        return binary_mask

    def fine_tune(
            self,
            train_data: Dataset,
            original_input_size: int,
            criterion: torch.nn,
            optimizer: torch.optim,
            lr: float = 1e-5,
            num_epochs: int = 10
        ) -> None:
        """
        Fine-tunes the SAM model using the provided training.

        :param train_data: The training data to be used for fine-tuning.
        :param lr: The learning rate to be used for fine-tuning.
        :param num_epochs: The number of epochs to be used for fine-tuning.

        :return: None
        """
        # setting the model to training mode
        self.sam.train()

        # creating the training and validation data loaders
        train_loader = self.get_dataset(train_data)

        for epoch in range(num_epochs):
            epoch_losses = []
            for input_image, box_prompt, gt_mask in tqdm(train_loader):
                if box_prompt.shape[1] == 0:
                    continue

                # forward pass
                pred_mask = self.forward_pass(
                    input_image, box_prompt, original_input_size=original_input_size)

                # compute loss
                loss = criterion(pred_mask, gt_mask)

                # backward pass (compute gradients of parameters w.r.t. loss)
                optimizer.zero_grad()
                loss.backward()

                # optimize
                optimizer.step()
                epoch_losses.append(loss.item())

            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')
