import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import random
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================================
# Hyperparameters
# ================================
BATCH_SIZE = 128
EPOCHS = 200
INIT_LR = 0.1
WARMUP_EPOCHS = 5

mixup_alpha = 1.0
cutmix_alpha = 1.0
cutmix_prob = 0.5

# ================================
# MixUp + CutMix helper functions
# ================================
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    return x1, y1, x2, y2


def apply_mixup(inputs, targets):
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = inputs.size()[0]

    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def apply_cutmix(inputs, targets):
    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size).to(inputs.device)

    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)

    inputs_new = inputs.clone()
    inputs_new[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (inputs.size()[-1] * inputs.size()[-2]))
    targets_a, targets_b = targets, targets[index]
    return inputs_new, targets_a, targets_b, lam


# ================================
# Data Augmentation (SOTA)
# ================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ================================
# ResNet-18 for CIFAR-10
# ================================
def build_resnet18_cifar10():
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)

    return model


model = build_resnet18_cifar10().to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(model.parameters(), lr=INIT_LR,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS)


# ================================
# Warmup
# ================================
def warmup_lr(optimizer, epoch):
    if epoch < WARMUP_EPOCHS:
        lr = INIT_LR * (epoch + 1) / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# ================================
# Train & Test
# ================================
def train(epoch):
    model.train()
    warmup_lr(optimizer, epoch)

    total, correct, total_loss = 0, 0, 0
    t0 = time.time()

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        r = np.random.rand()
        if r < cutmix_prob:
            inputs, targets_a, targets_b, lam = apply_cutmix(inputs, targets)
        else:
            inputs, targets_a, targets_b, lam = apply_mixup(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = lam * criterion(outputs, targets_a) + \
            (1 - lam) * criterion(outputs, targets_b)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)

        # MixUp/CutMix accuracy参考论文写法（非真实acc）
        correct += (lam * pred.eq(targets_a).sum().item()
                    + (1 - lam) * pred.eq(targets_b).sum().item())

    print(f"Train Epoch {epoch+1:3d} | Loss: {total_loss/len(trainloader):.4f} "
          f"| Acc (approx): {100.*correct/total:.2f}% | Time: {time.time()-t0:.1f}s")


def test(epoch):
    model.eval()
    correct, total = 0, 0
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f"Test | Loss: {total_loss/len(testloader):.4f} "
          f"| Acc: {acc:.2f}% ({correct}/{total})")
    return acc


# ================================
# Main Loop
# ================================
best_acc = 0

for epoch in range(EPOCHS):
    train(epoch)
    acc = test(epoch)

    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "resnet18_mixup_cutmix_best.pth")
        print(f">>>> Saved new best model ({best_acc:.2f}%) <<<<")

    print("-" * 60)

print("Training finished. Best Acc:", best_acc)
