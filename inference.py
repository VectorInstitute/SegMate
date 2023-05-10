import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAM_Inference:
    def __init__(
        self,
        sam_checkpoint: str = "models/sam_vit_h.pth",
        model_type: str = "vit_h",
        device: str = "cuda",
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    ):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

    def segment(self, image_path:str)->None:
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self.mask_generator.generate(image)

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
        self.show_anns(masks, ax2)
        ax2.set_title('Image with Masks')
        ax2.axis('off')

        plt.savefig(os.path.join('outputs', image_path.split(os.path.sep)[-1]), bbox_inches='tight')
                
    def segment_ds(self, ds_path:str)->None:
        ds_name = ds_path.split(os.path.sep)[-1]
        ds_name = f"{ds_name}_with_masks"
        if not os.path.exists(os.path.join('outputs', ds_name)):
            os.makedirs(os.path.join('outputs', ds_name))

        data_transforms = transforms.Compose([
        transforms.ToTensor()
        ])

        # Load the dataset
        dataset = datasets.ImageFolder(root=ds_path, transform=data_transforms)

        # Create a data loader for the dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Iterate over the data loader to get batches of images
        for i, (images, _) in enumerate(tqdm(data_loader)):
            # Get the list of image names in the batch
            image_names = [dataset.samples[i + j][0] for j in range(len(images))]
        
            for i in range(len(images)):
                # Convert the PyTorch tensor to a numpy array in HWC format
                image = images[i].permute(1, 2, 0).numpy()

                # Convert the numpy array to uint8 format
                image = (image * 255).astype(np.uint8)

                masks = self.mask_generator.generate(image)
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
                self.show_anns(masks, ax2)
                ax2.set_title('Image with Masks')
                ax2.axis('off')

                # Get the subset and name of the image
                subset = image_names[i].split(os.path.sep)[-2]
                image_name = image_names[i].split(os.path.sep)[-1]

                # Save the mask to disk using the image name
                if not os.path.exists(os.path.join('outputs', ds_name, subset)):
                    os.makedirs(os.path.join('outputs', ds_name, subset))

                plt.savefig(os.path.join('outputs', ds_name, subset, image_name), bbox_inches='tight')
                plt.close(fig)

    @staticmethod
    def show_anns(anns: list, ax: plt.Axes) -> None:
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None)
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    sam = SAM_Inference()

    if args.image:
        sam.segment(image_path=args.image)
    elif args.dataset:
        sam.segment_ds(ds_path=args.dataset)