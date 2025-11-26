# cifar10_pytorch_lab.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
# 运行指令：CUDA_VISIBLE_DEVICES=2 python LeNet-5.py
# ==============================================================================
# 1. 配置与准备 (Configuration and Setup)
# ==============================================================================

# 设置设备 (GPU优先, 否则CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"INFO: Using device: {DEVICE}")

# 定义超参数
BATCH_SIZE = 128  # 可以尝试不同的批大小，如32, 64, 128
LEARNING_RATE = 0.001
EPOCHS_LENET = 10
EPOCHS_OPTIMIZED = 100

# CIFAR-10 类别
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ==============================================================================
# 2. 数据加载与预处理 (Data Loading and Preprocessing)
# 对应 MindSpore 手册 2.5.3.5 节
# ==============================================================================

print("INFO: Preparing CIFAR-10 dataset...")
# 训练集的数据增强与预处理
# 对应文档中的随机裁剪、随机翻转、归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5)
])


# 测试集的预处理 (无需数据增强)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集，torchvision会自动下载
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print("INFO: Dataset prepared successfully.")


# ==============================================================================
# 3. 辅助函数 (Utility Functions)
# ==============================================================================

# 显示图像的函数
def imshow(img):
    """反标准化并显示图像"""
    # 反标准化: pixel = normalized_pixel * std + mean
    img[0] = img[0] * 0.2023 + 0.4914
    img[1] = img[1] * 0.1994 + 0.4822
    img[2] = img[2] * 0.2010 + 0.4465
    npimg = img.numpy()
    # Matplotlib 需要维度顺序为 [H, W, C]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 可视化预测结果的函数
def visualize_predictions(model, loader, num_images=10):
    """可视化模型对一些测试图像的预测结果"""
    print("INFO: Visualizing model predictions...")
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 8))

    with torch.no_grad():
        dataiter = iter(loader)
        images, labels = next(dataiter)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(images.size(0)):
            if images_so_far >= num_images:
                break
            images_so_far += 1
            ax = plt.subplot(2, 5, images_so_far)
            ax.axis('off')
            
            # 判断预测是否正确，用颜色区分
            color = "green" if preds[i] == labels[i] else "red"
            ax.set_title(f'pred: {CLASSES[preds[i]]}\n(true: {CLASSES[labels[i]]})', color=color)
            
            # 反标准化并显示图像
            img_display = images[i].cpu().clone()
            img_display[0] = img_display[0] * 0.2023 + 0.4914
            img_display[1] = img_display[1] * 0.1994 + 0.4822
            img_display[2] = img_display[2] * 0.2010 + 0.4465
            plt.imshow(np.transpose(img_display.numpy(), (1, 2, 0)))
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 4. 模型定义 (Model Definitions)
# ==============================================================================

# --- 原始 LeNet-5 模型 ---
# 对应 MindSpore 手册 2.5.4 节
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        # 输入维度计算:
        # 初始: 32x32 -> conv1(5x5) -> 28x28 -> pool -> 14x14
        # -> conv2(5x5) -> 10x10 -> pool -> 5x5
        # 展平后维度: 16通道 * 5 * 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 优化版模型 ---
# 对应 MindSpore 手册 2.5.6 节
class LeNet5_Optimized(nn.Module):
    def __init__(self):
        super(LeNet5_Optimized, self).__init__()
        # 使用 nn.Sequential 简化结构
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)#32*32->16*16
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)#16*16->8*8
        )
        # 初始: 32x32 -> pool1 -> 16x16 -> pool2 -> 8x8
        # 展平后维度: 128通道 * 8 * 8
        self.fc_block = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
        return x


# ==============================================================================
# 5. 训练与测试逻辑 (Training and Testing Logic)
# 对应 MindSpore 手册 2.5.5 节
# ==============================================================================

# 训练函数
def train(epoch, model, loader, optimizer, criterion):
    model.train() # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    end_time = time.time()
    epoch_time = end_time - start_time
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

# 测试函数
def test(model, loader, criterion):
    model.eval() # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    print(f"Test Results | Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}% ({correct}/{total})")
    print("-" * 50)
    return accuracy

# ==============================================================================
# 6. 主执行流程 (Main Execution Flow)
# ==============================================================================

if __name__ == "__main__":
    # --- 预览数据 ---
    print("\nINFO: Displaying some random training images...")
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:4])) # 显示4张图片
    print('Labels:', ' '.join(f'{CLASSES[labels[j]]:5s}' for j in range(4)))
    
    # --- 阶段 1: 训练原始 LeNet-5 模型 ---
    print("\n" + "="*50)
    print("PHASE 1: Training the original LeNet-5 Model")
    print("="*50)
    
    lenet_model = LeNet5().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS_LENET):
        train(epoch, lenet_model, trainloader, optimizer, criterion)
        test(lenet_model, testloader, criterion)
        
    print("INFO: LeNet-5 training finished.")
    torch.save(lenet_model.state_dict(), 'lenet5_cifar10.pth')
    
    # 可视化原始模型的预测结果
    visualize_predictions(lenet_model, testloader)

    # --- 阶段 2: 训练优化版模型 ---
    print("\n" + "="*50)
    print("PHASE 2: Training the Optimized Model")
    print("="*50)

    optimized_model = LeNet5_Optimized().to(DEVICE)
    criterion_opt = nn.CrossEntropyLoss()
    optimizer_opt = optim.Adam(optimized_model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.88
    for epoch in range(EPOCHS_OPTIMIZED):
        train(epoch, optimized_model, trainloader, optimizer_opt, criterion_opt)
        acc = test(optimized_model, testloader, criterion_opt)
        
        # 保存最佳模型
        if acc > best_acc:
            print(f"INFO: New best accuracy! Saving model to optimized_model_cifar10.pth")
            torch.save(optimized_model.state_dict(), 'optimized_model_cifar10.pth')
            best_acc = acc
            
    print(f"INFO: Optimized model training finished. Best Test Accuracy: {best_acc:.2f}%")
    
    # 加载最佳模型并可视化预测结果
    best_model = LeNet5_Optimized().to(DEVICE)
    best_model.load_state_dict(torch.load('optimized_model_cifar10.pth'))
    visualize_predictions(best_model, testloader)

    print("\n" + "="*50)
    print("Experiment Complete.")
    print("="*50)