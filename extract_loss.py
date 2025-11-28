import json
import re
import matplotlib.pyplot as plt
import os

notebook_path = '/home2/lihaoyu/ML/DeepLabv3.ipynb'
output_image_path = '/home2/lihaoyu/ML/deeplab_loss_curve.png'

data = []

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output['output_type'] == 'stream' and 'text' in output:
                    # text can be a list of strings
                    lines = output['text']
                    if isinstance(lines, str):
                        lines = [lines]
                    for line in lines:
                        # Look for pattern: Epoch [1/100], Loss: 0.6678, LR: 0.009998
                        match = re.search(r'Epoch \[(\d+)/(\d+)\], Loss: ([\d\.]+), LR: ([\d\.]+)', line)
                        if match:
                            epoch = int(match.group(1))
                            loss = float(match.group(3))
                            lr = float(match.group(4))
                            data.append((epoch, loss, lr))

    if not data:
        print("No loss data found!")
        exit(1)

    # Sort by epoch just in case
    data.sort(key=lambda x: x[0])

    epochs = [x[0] for x in data]
    losses = [x[1] for x in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.title('DeepLabv3 Training Loss Curve on PASCAL VOC2012', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Loss curve saved to {output_image_path}")
    print(f"Extracted {len(data)} data points.")

except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
