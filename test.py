import torch
import torch.nn as nn
import torch.nn.functional as F #!!!!!
import torch.utils.checkpoint as checkpoint # 用于节省显存的技巧
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# -----------------------------------------------------------------------------
# 核心辅助函数
# -----------------------------------------------------------------------------

def window_partition(x, window_size):
    """
    将特征图分割成非重叠的窗口。
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口合并回特征图。
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始特征图高度
        W (int): 原始特征图宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# -----------------------------------------------------------------------------
# 核心组件 1: 窗口自注意力 (Window Attention)
# -----------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """ 基于窗口的多头自注意力模块 (W-MSA) """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏置 (Relative Position Bias)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)) 

        # 用于计算相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # 计算相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (nW, N, N) 或 None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C/nH

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # QK^T

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 添加移位窗口的 mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -----------------------------------------------------------------------------
# 核心组件 2: Swin Transformer 块
# -----------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer 块，包含 W-MSA 和 SW-MSA (移位窗口) """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 确定是否进行移位操作
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def create_mask(self, H, W):
        """ 创建用于移位窗口的 attention mask """
        if self.shift_size > 0:
            # H, W 必须能被 window_size 整除
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征图大小不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 1. 循环移位 (Cyclic Shift)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            shifted_x = x
            attn_mask = None

        # 2. W-MSA/SW-MSA
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, N, C

        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, N, C

        # 3. 反转窗口，反转移位
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # 4. FFN (Feed Forward Network)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x) # 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        
        return x

# -----------------------------------------------------------------------------
# 核心组件 3: Patch Merging (下采样)
# -----------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    下采样层，将特征图分辨率减半，通道数翻倍。
    例如： H/2 x W/2 x 2C
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 2x2 patch 的 4 个子区域拼接 -> 4C 维度
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) 
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)

        # 将 H x W 划分成 4 个子区域
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B L/4 4*C

        x = self.norm(x)
        x = self.reduction(x) # 降维到 2C

        return x

# -----------------------------------------------------------------------------
# 核心组件 4: Patch Embedding (初始嵌入)
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """ 图像到 Patch Embeddings 的初始转换 """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 使用一个卷积层实现 Patch 划分和线性嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 确保输入图像尺寸是 Patch Size 的整数倍
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B C H W -> B C Ph*Pw -> B Ph*Pw C (B L C)
        if self.norm is not None:
            x = self.norm(x)
        return x

# -----------------------------------------------------------------------------
# 核心组件 5: Swin Transformer 整体模型
# -----------------------------------------------------------------------------

class BasicLayer(nn.Module):
    """ Swin Transformer 中的一个基本阶段 (Stage) """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 堆叠 Swin Transformer 块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样层 (除最后一阶段外，每阶段末尾都进行)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                # 使用 checkpoint 减少反向传播时的显存占用
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    """ Swin Transformer 模型的主体结构 """
    def __init__(self, img_size=32, patch_size=2, in_chans=3, num_classes=10,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # 1. 初始 Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置编码 (可选，Swin-T 默认不使用)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. 差分 DropPath 比率 (用于每个 Block)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 3. 构建 Basic Layers (Stage 1-4)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2**i_layer),
                               input_resolution=(patches_resolution[0] // (2**i_layer),
                                                 patches_resolution[1] // (2**i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # 4. 最终分类层 (与 CNN 中的 GAP 类似)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 对 L 维度取平均
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        # 可选：添加绝对位置编码
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Swin Transformer Layers
        for layer in self.layers:
            x = layer(x)

        # Global Average Pooling (L, C) -> (1, C)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) # B C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# -----------------------------------------------------------------------------
# 模型实例化与运行
# -----------------------------------------------------------------------------

# 假设输入是 32x32 RGB 图像，10 个类别 (如 CIFAR-10)
INPUT_IMAGE_SIZE = 32
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Swin Transformer Tiny (Swin-T) 配置，调整 window_size 和 patch_size 以适应 32x32 输入
# 原始 Swin-T 使用 224x224, window_size=7, patch_size=4
swin_config = {
    "img_size": INPUT_IMAGE_SIZE,
    "patch_size": 2,          # 32x32 -> 16x16 初始 tokens
    "in_chans": 3,
    "num_classes": NUM_CLASSES,
    "embed_dim": 96,          # C = 96
    "depths": [2, 2, 6, 2],   # 4 个阶段的 Blocks 数量
    "num_heads": [3, 6, 12, 24], # 4 个阶段的 Head 数量
    "window_size": 4,         # 适应 16x16, 8x8 等较小的特征图
    "drop_path_rate": 0.2     # 相对较高的 DropPath 率，防止过拟合
}

# 实例化 Swin Transformer 模型
swin_net = SwinTransformer(**swin_config).to(device)

print(f"--- Swin Transformer ({swin_config['img_size']}x{swin_config['img_size']}) 结构 ---")
print(swin_net)

# 打印参数总量 (Swin-T 约为 28M，这个版本会更小)
print(f"\nSwin Transformer 可训练参数总量: {sum(p.numel() for p in swin_net.parameters() if p.requires_grad)/1e6:.2f} M")

# 示例：使用新的模型和 AdamW 优化器
# Swin Transformer 论文中推荐使用 AdamW 优化器和余弦学习率衰减
optimizer_swin = optim.AdamW(swin_net.parameters(), lr=5e-4, weight_decay=0.05)

# 模拟训练步骤 (你需要自行填充 train/test 函数逻辑)
# num_epochs_opt = 100 
# for epoch in range(num_epochs_opt):
#     train(epoch, swin_net, trainloader, optimizer_swin) 
#     test(swin_net, testloader)