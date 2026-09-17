import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --------------------------------------------------
# RETFound
# --------------------------------------------------

sys.path.insert(0, "/content/RETFound_MAE")
import models_vit

# --------------------------------------------------
# Paths
# --------------------------------------------------

CHECKPOINT_PATH = "/content/drive/MyDrive/REFT/checkpoint-best.pth"

IMAGE_PATH = "/content/SULYAP-Image-Classification-1/test/Frisen Grade 0/342_jpg.rf.50af5b8fda39aee01dbd07bdefb5ff32.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# --------------------------------------------------
# Model
# --------------------------------------------------

NUM_CLASSES = 5

model = models_vit.vit_large_patch16(
    num_classes=NUM_CLASSES,
    drop_path_rate=0.1,
    global_pool=True
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

# Remove module. prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key.replace("module.", "", 1)

    clean_state_dict[key] = value

msg = model.load_state_dict(
    clean_state_dict,
    strict=False
)

print("Checkpoint loaded:")
print(msg)

model = model.to(device)
model.eval()

# --------------------------------------------------
# Wrapper
# --------------------------------------------------

class RETFoundWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


wrapped_model = RETFoundWrapper(model)
wrapped_model.eval()

# --------------------------------------------------
# Transform
# --------------------------------------------------

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])

# --------------------------------------------------
# Load image
# --------------------------------------------------

pil_img = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = transform(pil_img)

input_tensor = image_tensor.unsqueeze(0).to(device)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    logits = wrapped_model(input_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

print()
print("Logits:", logits)
print("Prediction:", prediction.item())
print("Confidence:", confidence.item())

# --------------------------------------------------
# Grad-CAM++ reshape
# --------------------------------------------------

def reshape_transform(tensor):

    # Remove CLS token
    tensor = tensor[:, 1:, :]

    # 224 / 16 = 14
    h = 14
    w = 14

    tensor = tensor.reshape(
        tensor.size(0),
        h,
        w,
        tensor.size(2)
    )

    tensor = tensor.permute(
        0,
        3,
        1,
        2
    )

    return tensor

# --------------------------------------------------
# Target layer
# --------------------------------------------------

target_layer = wrapped_model.model.blocks[-1].norm1

print()
print("Target layer:")
print(target_layer)

# --------------------------------------------------
# Grad-CAM++
# --------------------------------------------------

cam = GradCAMPlusPlus(
    model=wrapped_model,
    target_layers=[target_layer],
    reshape_transform=reshape_transform
)

target = [
    ClassifierOutputTarget(
        prediction.item()
    )
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=target
)[0]

# --------------------------------------------------
# Original image
# --------------------------------------------------

rgb_img = np.array(
    pil_img.resize((224, 224))
).astype(np.float32) / 255.0

# --------------------------------------------------
# Generate visualization
# --------------------------------------------------

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

# --------------------------------------------------
# Display
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(rgb_img)

plt.title("Original")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(visualization)

plt.title(
    "RETFound Grad-CAM++\n"
    "Prediction: {}\n"
    "Confidence: {:.2f}%".format(
        prediction.item(),
        confidence.item() * 100
    )
)

plt.axis("off")

plt.tight_layout()

plt.show()