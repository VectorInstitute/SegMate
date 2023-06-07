from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
import dataset as ds
import utils
import monai
import torch
import cv2


class SAMkit:
    """
    Class for interacting with the Segment Anything Model (SAM) for image segmentation.

    Attributes:
        model_type (str): The type of SAM: (vit_b, vit_l, vit_h) -> b: base, l: large, h: huge.
        checkpoint (str): The path to the pre-trained SAM.
        device (str):  The device to load the model on (default: cuda).

    Methods:
        __init__(self, model_path):
            Initializes the SAMkit object with the provided model path and model type.

        load_image(self, image):
            Loads the image.
        
        save_mask(self, binary_mask, output_path):
            Saves the resulting segmentation mask to the specified output path.

        preprocess_input(self, image, bbox_prompt):
            Transform and preprocesses the input image and bounding boxes to the required size and converts them to tensors.

        encode_input(self, input_image, bbox_prompt):
            Encodes the input image and bounding boxes.

        postprocess_mask(self, low_res_masks, transformed_input_size, original_input_size):
            Post-processes the segmentation mask to the original input size.

        generate_mask(self, image_embedding, sparse_embeddings, dense_embeddings, multimask_output=False):
            Generates the segmentation mask.

        segment(self):
            Performs image segmentation using the loaded image and input prompt.

        auto_segment(self, image, bbox_prompt, output_path, multimask_output=False):
            Performs image segmentation using the provided image and input prompt.

        visualize(self, image, bbox_prompt, output_path, multimask_output=False):
            Visualizes the segmentation mask.

        get_bis_dataset(self, dataset, preprocess, img_size, device):
            Returns the Building Image Segmentation (BIS) Dataset.

        forward(self, input_image, bbox_prompt):
            Performs a forward pass on the SAM model.
        
        fine_tune(self, train_data, val_data, num_epochs=10, batch_size=16):
            Fine-tunes the SAM model using the provided training and validation data.

    """

    def __init__(self, model_type='vit_b', checkpoint='sam_vit_b_01ec64.pth', device='cuda'):
        """
        Initializes the SAMkit object with the provided model path.

        Args:
            model_path (str): The path to the pre-trained SAM model.
        """
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
    
    def load_image(self, image):
        """
        Loads the image.

        Args:
            image (str): The path to the image.
        
        Returns:
            image (numpy.ndarray): The loaded image.
        """
        # loading the image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    def show_image(self, image):
        """
        Shows the image.

        Args:
            image (numpy.ndarray): The image to be shown.
        """
        # showing the image
        plt.imshow(image)
        ax.axis('off')
        plt.show()

    def save_mask(self, binary_mask, output_path):
        """
        Saves the resulting segmentation mask to the specified output path.

        Args:
            output_path (str): The path to save the segmentation map.
        """
        # saving the segmentation mask
        cv2.imwrite(output_path, binary_mask)

    def preprocess_input(self, image, bbox_prompt):
        """
        Transform and preprocesses the input image and bounding boxes to the required size and converts them to tensors.

        Args:
            image (numpy.ndarray): The image to be transformed.
            bbox_prompt (numpy.ndarray): The bounding boxes to be transformed.
        
        Returns:
            image (torch.Tensor): The transformed image.
            bbox_prompt (torch.Tensor): The transformed bounding boxes.
        """
        # transforming the image to the required size and preprocess it
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image = torch.as_tensor(input_image, device=self.device)
        input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.sam.preprocess(input_image)

        # transforming the bounding boxes to the required size
        bbox_prompt = transform.apply_boxes(bbox_prompt, image.shape[:2])
        bbox_prompt = torch.as_tensor(bbox_prompt, dtype=torch.float, device=self.device)

        return image, bbox_prompt
    
    def encode_input(self, input_image, bbox_prompt):
        """
        Encodes the input image and bounding boxes.

        Args:
            image (torch.Tensor): The transformed image.
            bbox_prompt (torch.Tensor): The transformed bounding boxes.
        
        Returns:
            image_embedding (torch.Tensor): The encoded image.
            sparse_embeddings (torch.Tensor): The encoded sparse bounding boxes.
            dense_embeddings (torch.Tensor): The encoded dense bounding boxes.
        """
        # encoding the image with sam's image encoder
        image_embedding = self.sam.image_encoder(input_image)

        # encoding the prompt with sam's prompt encoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
        points=None,
        boxes=bbox_prompt,
        masks=None,
        )

        return image_embedding, sparse_embeddings, dense_embeddings

    def postprocess_mask(self, low_res_masks, transformed_input_size, original_input_size):
        """
        Post-processes the segmentation mask to the original input size.

        Args:
            binary_mask (torch.Tensor): The binarized segmentation mask.
            original_input_size (tuple): The original input size of the image.
        
        Returns:
            binary_mask (numpy.ndarray): The binarized segmentation mask of the image.
        """
        # post-processing the segmentation mask
        upscaled_masks = self.sam.postprocess_masks(low_res_masks, transformed_input_size, original_input_size)
        thresholded_mask =  F.threshold(upscaled_masks, 0.0, 0)
        binary_mask = F.normalize(thresholded_mask)
        binary_mask = binary_mask.sum(axis=0).cpu().numpy()

        return binary_mask
    
    def generate_mask(self, image_embedding, sparse_embeddings, dense_embeddings, multimask_output=False):
        """
        Generates the segmentation mask.

        Args:
            image_embedding (torch.Tensor): The encoded image.
            sparse_embeddings (torch.Tensor): The encoded sparse bounding boxes.
            dense_embeddings (torch.Tensor): The encoded dense bounding boxes.
        
        Returns:
            low_res_masks (torch.Tensor): The generated segmentation mask.
            iou_predictions (torch.Tensor): The generated IOU predictions.
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

    def segment(self, image, prompt, output_path=None):
        """
        Performs image segmentation using the loaded image and input prompt.
        
        Args:
            image (str): The image or the path to the image to be segmented.
            prompt (str) or (numpy.ndarray): The prompt to be used for segmentation. The prompt can be a text or a list of bounding boxes.
            output_path (str): The path to save the resulting segmentation mask.
            
        Returns:
            binary_mask (numpy.ndarray): The binarized segmentation mask of the image.
        """
        # setting the model to evaluation mode
        self.sam.eval()

        # loading the image if the input is a path to an image
        if isinstance(image, str):
            image = self.load_image(image)
        
        #  converting the text prompt to a list of bounding boxes with a zero-shot object detection model (Grounding Dino, etc.)
        if isinstance(prompt, str):
            # To be implemented
            pass
        else:
            # converting the bounding boxes from [x_min, y_min, width, height] format to [x_min, y_min, x_max, y_max] format
            bbox_prompt = utils.convert_bboxes(prompt)

        # performing image segmentation with sam
        with torch.no_grad():
            # preprocessing and transforming the image and bounding box prompt(s) with sam's functions
            input_image, bbox_prompt = self.preprocess_input(image, bbox_prompt)
            
            # encoding the image and the prompt with sam's encoders
            image_embedding, sparse_embeddings, dense_embeddings = self.encode_input(input_image, bbox_prompt)

            # generating the segmentation mask from the image and the prompt embeddings
            low_res_masks, iou_predictions = self.generate_mask(self, image_embedding, sparse_embeddings, dense_embeddings, multimask_output=False)

            # postprocessing the segmentation mask and converting it to a numpy array
            binary_mask = self.postprocess_mask(low_res_masks, transformed_input_size=tuple(input_image.shape[-2:]), original_input_size=image.shape[:2])
        
        # saving the segmentation mask if the output path is provided
        if output_path is not None:
            self.save_mask(binary_mask, output_path)

        return binary_mask
    

    def auto_segment(self, 
                     image, 
                     points_per_side=32,
                     pred_iou_thresh=0.86,
                     stability_score_thresh=0.92,
                     crop_n_layers=1,
                     crop_n_points_downscale_factor=2,
                     min_mask_region_area=100):
        """
        Performs image segmentation using the automatic mask generation method.
        
        Args:
            image (str): The image or the path to the image to be segmented.
            points_per_side (int): The number of points per side of the mask.
            pred_iou_thresh (float): The IOU threshold for the predicted mask.
            stability_score_thresh (float): The stability score threshold for the predicted mask.
            crop_n_layers (int): The number of layers to crop from the image.
            crop_n_points_downscale_factor (int): The downscale factor for the number of points to crop from the image.
            min_mask_region_area (int): The minimum area of the mask region.
            
        Returns:
            masks (numpy.ndarray): The generated segmentation mask of the image.
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
            image = self.load_image(image)

        masks = mask_generator.generate(image)

        return masks
    
    def visualize_automask(self, image, masks, output_path=None):
        """
        Visualizes the segmentation mask on the image and saves the resulting visualization.

        Args:
            image (str): The image or the path to the image to be segmented.
            mask (numpy.ndarray): The segmentation mask to be visualized.
            output_path (str): The path to save the resulting visualization.
        """
        # loading the image if the input is a path to an image
        if isinstance(image, str):
            image = self.load_image(image)

        # Create a new figure with two axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

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

    
    def get_bis_dataset(self, dataset):
        """
        Prepare the data of the desired set.
        
        Args:
            dataset (HuggingFace dataset): The set of the data to be prepared.

        Returns:
            train_dataloader (torch.utils.data.DataLoader): The data loader of the desired set.
        """
        # creating the dataset
        train_dataset = ds.BISDataset(dataset=dataset, preprocess=self.sam.preprocess, img_size=self.sam.image_encoder.img_size, device=self.device)

        # creating the data loader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

        return train_dataloader

    def forward_pass(self, image, bbox_prompt):
        """
        Performs a forward pass on the image and the prompt.
        """
        # preprocessing and transforming the image and bounding box prompt(s) with sam's functions
        input_image, bbox_prompt = self.preprocess_input(image, bbox_prompt)
        
        # encoding the image and the prompt with sam's encoders
        image_embedding, sparse_embeddings, dense_embeddings = self.encode_input(input_image, bbox_prompt)

        # generating the segmentation mask from the image and the prompt embeddings
        low_res_masks, _ = self.generate_mask(self, image_embedding, sparse_embeddings, dense_embeddings, multimask_output=False)

        # postprocessing the segmentation mask and converting it to a numpy array
        binary_mask = self.postprocess_mask(low_res_masks, transformed_input_size=tuple(input_image.shape[-2:]), original_input_size=image.shape[:2])

        return binary_mask

    def fine_tune(self, train_data, lr=1e-5, num_epochs=10):
        """
        Fine-tunes the SAM model using the provided training.
        """
        # setting the model to training mode
        self.sam.train()

        # creating the training and validation data loaders
        train_loader = self.get_bis_dataset(train_data)

        # creating the optimizer and the loss function
        optimizer = torch.optim.Adam(self.sam.mask_decoder.parameters(), lr=lr)
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        for epoch in range(num_epochs):
            epoch_losses = []
            for input_image, box_prompt, gt_mask in tqdm(train_loader):
                if box_prompt.shape[1] == 0:
                    continue

                # forward pass
                pred_mask = self.forward_pass(input_image, box_prompt)

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