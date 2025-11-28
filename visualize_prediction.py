import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
NUM_CLASSES = 21
MODEL_PATH = 'models/deeplabv3_voc.pth'
IMAGE_PATH = 'data/VOC2012/JPEGImages/2007_006449.jpg'
MASK_PATH = 'data/VOC2012/SegmentationClass/2007_006449.png'
OUTPUT_PATH = 'deeplab_prediction_detail.png'
DEVICE = torch.device("cpu")

# Define VOC Colormap
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=np.uint8)

def create_model(num_classes):
    """
    Recreate the DeepLabv3 model architecture used in training.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False, aux_loss=True)
    model.classifier = DeepLabHead(2048, num_classes)
    return model

def visualize_prediction():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print("Loading model...")
    model = create_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # 2. Load Image and Mask
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
        print(f"Error: Image or Mask file not found.")
        return

    print(f"Processing image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert('RGB')
    
    # Preprocessing
    input_image = F.to_tensor(image) * 255.0
    input_image = F.normalize(input_image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
    input_image = input_image.unsqueeze(0).to(DEVICE)

    # 3. Prediction
    print("Running inference...")
    with torch.no_grad():
        output = model(input_image)['out'][0]
    
    pred_mask = torch.argmax(output, dim=0).cpu().numpy()
    pred_mask_colored = VOC_COLORMAP[pred_mask]

    # 4. Process Ground Truth
    true_mask_pil = Image.open(MASK_PATH)
    true_mask_np = np.array(true_mask_pil)
    # Handle ignore label (255) by mapping it to background (0) for visualization purposes
    # or keep it as is if colormap handles it (VOC_COLORMAP is size 21, index 255 would crash)
    true_mask_np[true_mask_np == 255] = 0
    true_mask_colored = VOC_COLORMAP[true_mask_np]

    # 5. Visualization
    print("Generating visualization...")
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image", fontsize=16)
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask", fontsize=16)
    plt.imshow(pred_mask_colored)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth", fontsize=16)
    plt.imshow(true_mask_colored)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Prediction detail saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    visualize_prediction()
