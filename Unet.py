# ======================================================================
# 1. IMPORTS & CONFIGURATION
# ======================================================================
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from torch.nn import functional as F

# --- Main Configuration Class ---
class Config:
    GPU_ID = 3
    DEVICE = None
    DATA_PATH_PREFIX = 'data/Unet/'
    TRAIN_IMG_PATH = os.path.join(DATA_PATH_PREFIX, 'train-volume.tif')
    TRAIN_LBL_PATH = os.path.join(DATA_PATH_PREFIX, 'train-labels.tif')
    MODEL_SAVE_PATH = 'best_transunet_model.pth'
    
    # --- Training Hyperparameters ---
    NUM_EPOCHS = 150
    BATCH_SIZE = 2 # TransUNet is memory-intensive
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15
    
    # --- Model Specific Parameters for TransUNet ---
    IMG_DIM = 512 # Input images will be resized to this
    N_CHANNELS = 1
    N_CLASSES = 2
    
    # --- ViT Parameters ---
    VIT_PATCH_SIZE = 16
    VIT_HIDDEN_DIM = 768
    VIT_MLP_DIM = 3072
    VIT_NUM_HEADS = 12
    VIT_NUM_LAYERS = 12

# ======================================================================
# 2. UTILITIES & SETUP
# ======================================================================
def setup_environment(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_ID)
    config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {config.DEVICE}")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_multipage_tiff(path):
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

config = Config()
setup_environment(config)

# ======================================================================
# 3. ADVANCED DATA AUGMENTATION (ALBUMENTATIONS)
# ======================================================================
class ISBI_Dataset(Dataset):
    def __init__(self, images, labels, img_dim, augment=False):
        self.images = images
        self.labels = labels
        self.img_dim = img_dim
        self.augment = augment
        
        # Define base transforms (resizing and tensor conversion)
        self.base_transform = A.Compose([
            A.Resize(img_dim, img_dim, interpolation=Image.BILINEAR),
            A.Normalize(mean=(0.5,), std=(0.5,)), # Normalize to [-1, 1]
            ToTensorV2(),
        ])
        
        # Define augmentation pipeline if enabled
        if self.augment:
            self.aug_transform = A.Compose([
                A.Resize(img_dim, img_dim, interpolation=Image.BILINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # SOTA augmentations for medical images
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
                # Color augmentations
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx]
        mask[mask == 255] = 1 # Convert mask to 0s and 1s

        if self.augment:
            transformed = self.aug_transform(image=image, mask=mask)
        else:
            transformed = self.base_transform(image=image, mask=mask)
            
        return transformed['image'].float(), transformed['mask'].long()

# ======================================================================
# 4. SOTA MODEL: TransUNet
# ======================================================================
# --- Helper Modules for TransUNet ---
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

# --- Main TransUNet Architecture ---
class TransUNet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_patch_size, vit_hidden_dim, vit_mlp_dim,
                 vit_num_heads, vit_num_layers):
        super().__init__()
        # --- CNN Encoder (Downsampling path) ---
        self.patch_dim = vit_patch_size
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2)
        
        # --- Bridge from CNN to Transformer ---
        self.to_patch_embedding = nn.Conv2d(256, vit_hidden_dim, kernel_size=self.patch_dim, stride=self.patch_dim)
        num_patches = (img_dim // (2**3 * self.patch_dim)) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, vit_hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_hidden_dim))

        # --- Transformer Encoder ---
        self.transformer = nn.Sequential(
            *[TransformerBlock(vit_hidden_dim, vit_num_heads, vit_hidden_dim // vit_num_heads, vit_mlp_dim) for _ in range(vit_num_layers)]
        )
        
        # --- Decoder (Upsampling Path) ---
        self.up3 = nn.ConvTranspose2d(vit_hidden_dim, 256, 2, stride=2)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x):
        # CNN Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        # Transformer
        patches = self.to_patch_embedding(p3)
        patches = rearrange(patches, 'b c h w -> b (h w) c')
        b, n, _ = patches.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        
        # Reshape to 2D for decoder
        transformer_out = x[:, 1:, :] # Drop CLS token
        H = W = int(np.sqrt(transformer_out.shape[1]))
        transformer_out = rearrange(transformer_out, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Decoder
        d3 = self.up3(transformer_out)
        d3 = torch.cat([F.interpolate(c3, d3.shape[2:]), d3], dim=1)
        d3 = self.dec_conv3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([F.interpolate(c2, d2.shape[2:]), d2], dim=1)
        d2 = self.dec_conv2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([F.interpolate(c1, d1.shape[2:]), d1], dim=1)
        d1 = self.dec_conv1(d1)
        
        # Final upsampling to original size
        out = self.final_conv(d1)
        return F.interpolate(out, size=(config.IMG_DIM, config.IMG_DIM), mode='bilinear', align_corners=False)


# ======================================================================
# 5. ADVANCED LOSS & METRICS
# ======================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred_logits, target):
        pred_prob = torch.softmax(pred_logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=config.N_CLASSES).permute(0, 3, 1, 2)
        pred_fg = pred_prob[:, 1, ...]
        target_fg = target_one_hot[:, 1, ...].float()
        intersection = (pred_fg * target_fg).sum()
        union = pred_fg.sum() + target_fg.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score

class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5):
        super().__init__()
        self.focal_weight = focal_weight
        self.focal = FocalLoss()
        self.dice = DiceLoss()
    def forward(self, pred, target):
        return self.focal_weight * self.focal(pred, target) + (1 - self.focal_weight) * self.dice(pred, target)

# Metric function remains the same
def dice_metric_func(pred_logits, target, smooth=1e-6):
    pred_class = pred_logits.argmax(dim=1)
    target_one_hot = F.one_hot(target, num_classes=config.N_CLASSES).permute(0, 3, 1, 2)
    pred_one_hot = F.one_hot(pred_class, num_classes=config.N_CLASSES).permute(0, 3, 1, 2)
    intersection = (pred_one_hot[:, 1] * target_one_hot[:, 1]).sum()
    union = pred_one_hot[:, 1].sum() + target_one_hot[:, 1].sum()
    return (2. * intersection + smooth) / (union + smooth)

# ======================================================================
# 6. TRAINING & EVALUATION LOOP
# ======================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice += dice_metric_func(outputs, masks).item()
    return total_dice / len(loader)

def evaluate_with_tta(model, image_tensor, device):
    """ Test-Time Augmentation for a single image """
    model.eval()
    image_tensor = image_tensor.to(device)
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0))
        predictions.append(torch.softmax(pred, dim=1))
        
    # Flipped
    with torch.no_grad():
        pred_hf = model(torch.flip(image_tensor, [2]).unsqueeze(0))
        predictions.append(torch.flip(torch.softmax(pred_hf, dim=1), [3]))
        
        pred_vf = model(torch.flip(image_tensor, [1]).unsqueeze(0))
        predictions.append(torch.flip(torch.softmax(pred_vf, dim=1), [2]))
        
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    return avg_pred.squeeze(0)

# ======================================================================
# 7. MAIN EXECUTION SCRIPT
# ======================================================================
def main():
    # --- Data Loading & Splitting ---
    train_images_full = load_multipage_tiff(config.TRAIN_IMG_PATH)
    train_labels_full = load_multipage_tiff(config.TRAIN_LBL_PATH)
    indices = list(range(len(train_images_full)))
    random.shuffle(indices)
    split_point = int(np.floor(config.VALIDATION_SPLIT * len(train_images_full)))
    val_indices, train_indices = indices[:split_point], indices[split_point:]

    train_dataset = ISBI_Dataset(train_images_full[train_indices], train_labels_full[train_indices], config.IMG_DIM, augment=True)
    val_dataset = ISBI_Dataset(train_images_full[val_indices], train_labels_full[val_indices], config.IMG_DIM, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Data ready: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # --- Model, Loss, Optimizer, Scheduler ---
    model = TransUNet(
        img_dim=config.IMG_DIM, in_channels=config.N_CHANNELS, classes=config.N_CLASSES,
        vit_patch_size=config.VIT_PATCH_SIZE, vit_hidden_dim=config.VIT_HIDDEN_DIM,
        vit_mlp_dim=config.VIT_MLP_DIM, vit_num_heads=config.VIT_NUM_HEADS,
        vit_num_layers=config.VIT_NUM_LAYERS
    ).to(config.DEVICE)
    criterion = CombinedFocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    # --- Training Loop ---
    best_val_dice = -1
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_dice': []}
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_dice = evaluate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> Best model saved with Dice: {best_val_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print(f"\nTraining finished in {(time.time() - start_time)/60:.2f} minutes.")

    # --- Final Visualization with TTA ---
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    num_visualize = min(5, len(val_dataset))
    for i in range(num_visualize):
        image, mask = val_dataset[i]
        
        # TTA Prediction
        pred_prob_tta = evaluate_with_tta(model, image, config.DEVICE)
        pred_mask_tta = torch.argmax(pred_prob_tta, dim=0).cpu().numpy()
        
        # Standard Prediction
        with torch.no_grad():
            pred_prob_std = model(image.to(config.DEVICE).unsqueeze(0))
            pred_mask_std = torch.argmax(pred_prob_std, dim=1).squeeze(0).cpu().numpy()

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1); plt.imshow(image.squeeze().numpy(), cmap='gray'); plt.title("Input Image"); plt.axis('off')
        plt.subplot(1, 4, 2); plt.imshow(mask.numpy(), cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(1, 4, 3); plt.imshow(pred_mask_std, cmap='gray'); plt.title("Prediction (Standard)"); plt.axis('off')
        plt.subplot(1, 4, 4); plt.imshow(pred_mask_tta, cmap='gray'); plt.title("Prediction (TTA)"); plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()