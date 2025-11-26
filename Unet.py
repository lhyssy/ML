# == 1. 导入与配置 (IMPORTS & CONFIG) ==
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from einops import rearrange

# --- 超参数配置 ---
class Config:
    GPU_ID = 3
    DEVICE = None
    DATA_PATH_PREFIX = 'data/Unet/'
    TRAIN_IMG_PATH = os.path.join(DATA_PATH_PREFIX, 'train-volume.tif')
    TRAIN_LBL_PATH = os.path.join(DATA_PATH_PREFIX, 'train-labels.tif')
    MODEL_SAVE_PATH = 'best_transunet_model.pth'
    NUM_EPOCHS = 150
    BATCH_SIZE = 2 # TransUNet模型较大，减小batch size
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    N_CHANNELS = 1
    N_CLASSES = 2
    EARLY_STOPPING_PATIENCE = 15

# --- 环境设置 ---
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

config = Config()
setup_environment(config)

# == 2. SOTA数据加载与增强 (ADVANCED DATA LOADING & AUGMENTATION) ==
def load_multipage_tiff(path):
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

class AlbumentationsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx].copy()
        mask[mask == 255] = 1 # 转换标签

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

# --- 定义增强流程 ---
# 注意：TransUNet通常在更大分辨率上预训练，这里我们调整图像大小以匹配
IMG_SIZE = 224
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Rotate(limit=35, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.Normalize(mean=(0.5,), std=(0.5,)), # 标准化到[-1, 1]
    ToTensorV2(),
])
val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# == 3. SOTA模型: TransUNet (MODEL: TransUNet) ==
# --- 辅助模块 ---
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, depths=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_stages = nn.ModuleList()
        for i in range(len(depths)):
            in_c = in_channels if i == 0 else depths[i-1]
            out_c = depths[i]
            stage = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
            self.encoder_stages.append(stage)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []
        for stage in self.encoder_stages:
            x = stage(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# --- TransUNet 主体 ---
class TransUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, img_size=224):
        super().__init__()
        # --- CNN Encoder ---
        self.encoder_depths = [64, 128, 256]
        self.cnn_encoder = CNNEncoder(in_channels, self.encoder_depths)
        
        # --- ViT Bottleneck ---
        self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        # 调整ViT的输入层以接受CNN的输出通道数
        self.vit.patch_embed.proj = nn.Conv2d(self.encoder_depths[-1], 768, kernel_size=16, stride=16)
        
        # --- Decoder ---
        self.decoder_depths = [512, 256, 128, 64]
        # 解码ViT输出的块
        self.up_vit = DecoderBlock(768, 512)
        
        self.decoder_blocks = nn.ModuleList()
        # 256->512, 128->256, 64->128
        in_depths = [d*2 for d in self.encoder_depths[::-1]] # [512, 256, 128]
        out_depths = self.encoder_depths[::-1] # [256, 128, 64]
        self.decoder_blocks.append(DecoderBlock(in_depths[0], out_depths[0]))
        self.decoder_blocks.append(DecoderBlock(in_depths[1], out_depths[1]))
        self.decoder_blocks.append(DecoderBlock(in_depths[2], out_depths[2]))
        
        self.final_conv = nn.Conv2d(out_depths[-1], n_classes, kernel_size=1)

    def forward(self, x):
        _, skips = self.cnn_encoder(x)
        vit_input = skips[-1]
        
        # --- ViT处理 ---
        # 手动将输入展平为ViT期望的序列格式
        # 输入vit_input [B, 256, 28, 28] -> 展平为 [B, 768, 1, 49] -> [B, 784, 768]
        # 这里为了简化，我们直接用patch_embed处理，它会处理成 [B, N, D]
        x_vit = self.vit.patch_embed(vit_input)
        x_vit = self.vit.pos_drop(x_vit)
        x_vit = self.vit.blocks(x_vit)
        x_vit = self.vit.norm(x_vit)
        
        # 将ViT输出 reshape 回2D图像格式
        # x_vit [B, N, D] -> [B, D, H, W]
        # N = (H/P)*(W/P)
        patch_size = self.vit.patch_embed.patch_size[0]
        h = w = int(vit_input.shape[2] / patch_size)
        x_vit = rearrange(x_vit, 'b (h w) c -> b c h w', h=h, w=w)

        # --- Decoder ---
        d_out = self.up_vit(x_vit, skips[-1]) # 这里的skip实际上是vit_input
        d_out = self.decoder_blocks[0](d_out, skips[-2])
        d_out = self.decoder_blocks[1](d_out, skips[-3])
        # 这里需要对输入x进行下采样以匹配第一个skip的尺寸
        x_resized_for_skip = nn.functional.interpolate(x, size=skips[0].shape[2:], mode='bilinear', align_corners=False)
        d_out = self.decoder_blocks[2](d_out, skips[0])
        
        # --- 上采样到原始尺寸 ---
        out = nn.functional.interpolate(d_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.final_conv(out)

# == 4. SOTA损失函数 (ADVANCED LOSS FUNCTION) ==
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_prob, target):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=config.N_CLASSES).permute(0, 3, 1, 2)
        pred_fg = pred_prob[:, 1, ...]
        target_fg = target_one_hot[:, 1, ...].float()
        intersection = (pred_fg * target_fg).sum()
        union = pred_fg.sum() + target_fg.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dice = self.dice_loss(probs, targets)
        return self.dice_weight * dice + (1 - self.dice_weight) * focal

# == 5. 训练与评估 (TRAINING & EVALUATION) ==
def dice_metric_func(pred_logits, target):
    pred_class = pred_logits.argmax(dim=1)
    # ... (dice calculation as before)
    return dice_score

# (train_one_epoch and evaluate functions remain largely the same, just update progress bar descriptions)

# == 6. 测试时增强 (TEST-TIME AUGMENTATION) ==
def tta_predict(model, image_tensor, device):
    """
    image_tensor: a single image tensor [C, H, W]
    """
    model.eval()
    
    transforms = {
        'original': lambda x: x,
        'hflip': lambda x: torch.flip(x, [2]),
    }
    
    predictions = []
    
    with torch.no_grad():
        for name, t in transforms.items():
            augmented_img = t(image_tensor.clone()).unsqueeze(0).to(device)
            output = model(augmented_img)
            
            # Reverse the transform for the prediction
            if name == 'hflip':
                output = torch.flip(output, [3])
            
            predictions.append(torch.softmax(output, dim=1))
            
    # Average the predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction.squeeze(0) # [C, H, W]

# == 7. 主程序 (MAIN EXECUTION) ==
def main():
    # --- Data Loading ---
    train_images = load_multipage_tiff(config.TRAIN_IMG_PATH)
    train_labels = load_multipage_tiff(config.TRAIN_LBL_PATH)

    indices = list(range(len(train_images)))
    random.shuffle(indices)
    split_point = int(np.floor(config.VALIDATION_SPLIT * len(train_images)))
    val_indices, train_indices = indices[:split_point], indices[split_point:]

    train_dataset = AlbumentationsDataset(train_images[train_indices], train_labels[train_indices], transform=train_transforms)
    val_dataset = AlbumentationsDataset(train_images[val_indices], train_labels[val_indices], transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    model = TransUNet(in_channels=config.N_CHANNELS, n_classes=config.N_CLASSES, img_size=IMG_SIZE).to(config.DEVICE)
    criterion = DiceFocalLoss().to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)


    visualize_with_tta(model, val_dataset, config)

def visualize_with_tta(model, dataset, config):
    print("\n--- Visualizing Predictions with TTA ---")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.to(config.DEVICE)
    
    for i in range(min(5, len(dataset))):
        image, mask = dataset[i] # image is already transformed
        
        # TTA Prediction
        pred_prob_tta = tta_predict(model, image, config.DEVICE)
        pred_mask_tta = torch.argmax(pred_prob_tta, dim=0).cpu().numpy()

        # Standard Prediction (for comparison)
        with torch.no_grad():
            output_std = model(image.unsqueeze(0).to(config.DEVICE))
            pred_mask_std = torch.argmax(output_std, dim=1).squeeze(0).cpu().numpy()

        plt.figure(figsize=(20, 5))
        # Denormalize image for visualization
        img_vis = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
        
        plt.subplot(1, 4, 1)
        plt.title("Input Image")
        plt.imshow(img_vis, cmap='gray')
        
        plt.subplot(1, 4, 2)
        plt.title("Ground Truth")
        plt.imshow(mask.numpy(), cmap='gray')
        
        plt.subplot(1, 4, 3)
        plt.title("Prediction (Standard)")
        plt.imshow(pred_mask_std, cmap='gray')
        
        plt.subplot(1, 4, 4)
        plt.title("Prediction (TTA)")
        plt.imshow(pred_mask_tta, cmap='gray')
        
        plt.show()

if __name__ == '__main__':
    main() # Remember to fill in the training loop