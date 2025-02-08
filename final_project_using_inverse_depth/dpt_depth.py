import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# Load DPT model and feature extractor
model_name = "Intel/dpt-large"
feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

def load_image(image_path):
    """ Load an image and convert it to RGB """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def predict_depth(image):
    """ Estimate depth using DPT model """
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return depth

def normalize_depth(depth):
    """ Normalize depth values for visualization """
    depth_min = depth.min()
    depth_max = depth.max()
    return (depth - depth_min) / (depth_max - depth_min)

def show_depth_map(image, depth):
    """ Display depth map alongside original image """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(normalize_depth(depth), cmap="magma")
    plt.title("Depth Map")
    plt.axis("off")

    plt.savefig('depth.png', bbox_inches ="tight")
    plt.show()


def show_images(image_list, caption_list, nsize=None):
    """ Display images in the list """
    plt.figure(figsize=(15, 5))


    num_images = len(image_list)
    if nsize is None:
       nrows, ncols = 1, num_images
    else:
        nrows, ncols = nsize[0], nsize[1]

    for j in range(num_images):
        plt.subplot(nrows, ncols, j+1)
        plt.imshow(image_list[j])
        plt.title(caption_list[j])
        plt.axis("off")

    plt.savefig('warping.png', bbox_inches ="tight")
    plt.show()


def get_dpt_depth(image_path):
    # Load and process both images
    image = load_image(image_path)

    # Predict depth
    depth = predict_depth(image)

    depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Show depth maps
    show_depth_map(image, depth)

    return image, depth