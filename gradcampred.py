import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# PATHS
# ============================================================

RETFOUND_PATH = "/content/RETFound_MAE"

TEST_DIR = "/content/SULYAP-Image-Classification-1/test"

CHECKPOINT_PATH = "/content/drive/MyDrive/REFT/checkpoint-best.pth"

OUTPUT_DIR = "/content/retfound_gradcam_results"

GRADCAM_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("RETFound + Grad-CAM++")
print("==========================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# RETFOUND IMPORT
# ============================================================

sys.path.insert(
    0,
    RETFOUND_PATH
)

import models_vit


# ============================================================
# FIND CLASSES
# ============================================================

class_names = sorted([
    folder
    for folder in os.listdir(TEST_DIR)
    if os.path.isdir(
        os.path.join(TEST_DIR, folder)
    )
])


print("Classes:")

for i, class_name in enumerate(class_names):

    print(
        "{} -> {}".format(
            i,
            class_name
        )
    )


NUM_CLASSES = len(class_names)

print()
print(
    "Number of classes:",
    NUM_CLASSES
)
print()


class_to_idx = {
    name: i
    for i, name in enumerate(class_names)
}

idx_to_class = {
    i: name
    for name, i in class_to_idx.items()
}


# ============================================================
# TRANSFORMATION
# ============================================================

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

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ============================================================
# CREATE RETFOUND MODEL
# ============================================================

print("Creating RETFound model...")


model = models_vit.vit_large_patch16(

    num_classes=NUM_CLASSES,

    drop_path_rate=0.1,

    global_pool=True
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint:")
print(CHECKPOINT_PATH)
print()


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)


if "model" in checkpoint:

    state_dict = checkpoint["model"]

else:

    state_dict = checkpoint


# Remove "module." prefix if present

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith("module."):

        key = key.replace(
            "module.",
            "",
            1
        )

    clean_state_dict[key] = value


state_dict = clean_state_dict


msg = model.load_state_dict(
    state_dict,
    strict=False
)


print("Checkpoint loading result:")
print(msg)
print()


model = model.to(device)

model.eval()


print("RETFound loaded successfully.")
print()


# ============================================================
# WRAPPER
# ============================================================

class RETFoundWrapper(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model


    def forward(self, x):

        return self.model(x)


wrapped_model = RETFoundWrapper(
    model
)

wrapped_model.eval()


# ============================================================
# GRAD-CAM++ RESHAPE
# ============================================================

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


# ============================================================
# TARGET LAYER
# ============================================================

target_layer = (
    wrapped_model
    .model
    .blocks[-1]
    .norm1
)


print("Grad-CAM target layer:")
print(target_layer)
print()


# ============================================================
# CREATE GRAD-CAM++
# ============================================================

cam = GradCAMPlusPlus(

    model=wrapped_model,

    target_layers=[
        target_layer
    ],

    reshape_transform=reshape_transform
)


# ============================================================
# FIND ALL IMAGES
# ============================================================

image_extensions = (

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".jfif"

)


image_paths = []


for class_name in class_names:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    for filename in sorted(
        os.listdir(class_dir)
    ):

        if filename.lower().endswith(
            image_extensions
        ):

            image_path = os.path.join(
                class_dir,
                filename
            )

            image_paths.append(
                (
                    image_path,
                    class_name
                )
            )


print(
    "Total test images:",
    len(image_paths)
)

print()


# ============================================================
# RESULT STORAGE
# ============================================================

results = []

actual_labels = []

predicted_labels = []


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

for image_number, (
    image_path,
    actual_class
) in enumerate(
    image_paths,
    start=1
):

    filename = os.path.basename(
        image_path
    )


    print(
        "[{}/{}] {}".format(
            image_number,
            len(image_paths),
            filename
        )
    )


    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        pil_img = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        image_tensor = transform(
            pil_img
        )


        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(device)
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            logits = wrapped_model(
                input_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        pred_index = prediction.item()

        pred_class = idx_to_class[
            pred_index
        ]

        confidence_value = (
            confidence.item()
        )


        # ----------------------------------------------------
        # GRAD-CAM++
        # ----------------------------------------------------

        target = [

            ClassifierOutputTarget(
                pred_index
            )

        ]


        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=target
        )[0]


        # ----------------------------------------------------
        # RGB IMAGE FOR VISUALIZATION
        # ----------------------------------------------------

        rgb_img = np.array(
            pil_img.resize(
                (224, 224)
            )
        ).astype(
            np.float32
        ) / 255.0


        # ----------------------------------------------------
        # GRAD-CAM OVERLAY
        # ----------------------------------------------------

        visualization = show_cam_on_image(

            rgb_img,

            grayscale_cam,

            use_rgb=True

        )


        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        actual_index = class_to_idx[
            actual_class
        ]


        actual_labels.append(
            actual_index
        )

        predicted_labels.append(
            pred_index
        )


        correct = (
            actual_class == pred_class
        )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append({

            "image": filename,

            "image_path": image_path,

            "actual": actual_class,

            "predicted": pred_class,

            "confidence": confidence_value,

            "correct": correct

        })


        # ----------------------------------------------------
        # CREATE FIGURE
        # ----------------------------------------------------

        plt.figure(
            figsize=(14, 6)
        )


        # Original

        plt.subplot(
            1,
            2,
            1
        )

        plt.imshow(
            rgb_img
        )

        plt.title(

            "Original\n"
            "Actual: {}".format(
                actual_class
            ),

            fontsize=14

        )

        plt.axis("off")


        # Grad-CAM++

        plt.subplot(
            1,
            2,
            2
        )

        plt.imshow(
            visualization
        )

        plt.title(

            "Grad-CAM++\n"
            "Predicted: {}\n"
            "Confidence: {:.2f}%".format(

                pred_class,

                confidence_value * 100

            ),

            fontsize=14

        )

        plt.axis("off")


        plt.tight_layout()


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        base_name = os.path.splitext(
            filename
        )[0]


        output_filename = (

            "{}_actual_{}_pred_{}.png".format(

                base_name,

                actual_class.replace(
                    " ",
                    "_"
                ),

                pred_class.replace(
                    " ",
                    "_"
                )

            )

        )


        output_path = os.path.join(

            GRADCAM_DIR,

            output_filename

        )


        plt.savefig(

            output_path,

            dpi=200,

            bbox_inches="tight"

        )


        plt.close()


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            "    Actual     :",
            actual_class
        )

        print(
            "    Predicted  :",
            pred_class
        )

        print(
            "    Confidence : {:.4f}".format(
                confidence_value
            )
        )

        print(
            "    Correct    :",
            correct
        )

        print()


    except Exception as e:

        print(
            "ERROR:",
            image_path
        )

        print(
            str(e)
        )

        print()

        continue


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    results
)


df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    actual_labels,

    predicted_labels

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    actual_labels,

    predicted_labels,

    labels=list(
        range(NUM_CLASSES)
    ),

    target_names=class_names,

    digits=4,

    zero_division=0

)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(
    "Images processed:",
    len(results)
)

print(
    "Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)

print()

print(report)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w"
) as f:

    f.write(
        "RETFound Classification Report\n"
    )

    f.write(
        "==========================================\n\n"
    )

    f.write(
        "Number of classes: {}\n".format(
            NUM_CLASSES
        )
    )

    f.write(
        "Images processed: {}\n".format(
            len(results)
        )
    )

    f.write(
        "Accuracy: {:.4f}\n\n".format(
            accuracy
        )
    )

    f.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    actual_labels,

    predicted_labels,

    labels=list(
        range(NUM_CLASSES)
    )

)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names

)


disp.plot(

    ax=ax,

    xticks_rotation=45

)


plt.title(
    "RETFound - Frisen Grade Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    CONFUSION_MATRIX_PATH,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("RESULTS SAVED")
print("==========================================")

print()
print("CSV:")
print(CSV_PATH)

print()
print("Grad-CAM++ images:")
print(GRADCAM_DIR)

print()
print("Classification report:")
print(REPORT_PATH)

print()
print("Confusion matrix:")
print(CONFUSION_MATRIX_PATH)

print()
print("==========================================")
print("DONE")
print("==========================================")