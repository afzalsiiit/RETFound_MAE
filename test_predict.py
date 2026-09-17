import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. RETFound path
# ============================================================

RETFOUND_PATH = '/content/RETFound_MAE'

sys.path.insert(0, RETFOUND_PATH)

import models_vit


# ============================================================
# 2. Paths
# ============================================================

TEST_DIR = ('/content/SULYAP-Image-Classification-1/test'
)

CHECKPOINT_PATH = ('/content/drive/MyDrive/REFT/checkpoint-best.pth'
)

OUTPUT_DIR = '/content/retfound_test_results'

IMAGE_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    'images'
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    'predictions.csv'
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    'confusion_matrix.png'
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    'classification_report.txt'
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    'summary.txt'
)


# ============================================================
# 3. Create output directories
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    IMAGE_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. Device
# ============================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print('==========================================')
print('RETFound Test & Prediction')
print('==========================================')
print('Device:', device)

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

print()


# ============================================================
# 5. Find classes automatically
# ============================================================

class_names = sorted([
    folder
    for folder in os.listdir(TEST_DIR)
    if os.path.isdir(
        os.path.join(TEST_DIR, folder)
    )
])

print('Classes found:')
for i, class_name in enumerate(class_names):
    print(i, '->', class_name)

NUM_CLASSES = len(class_names)

print()
print('Number of classes:', NUM_CLASSES)
print()


# ============================================================
# 6. Class mapping
# ============================================================

class_to_idx = {
    class_name: i
    for i, class_name in enumerate(class_names)
}

idx_to_class = {
    i: class_name
    for class_name, i in class_to_idx.items()
}


# ============================================================
# 7. Image preprocessing
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
# 8. Build RETFound classification model
# ============================================================

print('Creating RETFound model...')


model = models_vit.vit_large_patch16(
    num_classes=NUM_CLASSES,
    drop_path_rate=0.1,
    global_pool=True
)


# ============================================================
# 9. Load checkpoint
# ============================================================

print('Loading checkpoint:')
print(CHECKPOINT_PATH)
print()


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location='cpu'
)


# Some checkpoints have:
#
# checkpoint['model']
#
# while others may directly contain the state dictionary.

if 'model' in checkpoint:

    state_dict = checkpoint['model']

else:

    state_dict = checkpoint


# ============================================================
# 10. Remove "module." prefix if present
# ============================================================

new_state_dict = {}

for key, value in state_dict.items():

    if key.startswith('module.'):

        key = key.replace(
            'module.',
            '',
            1
        )

    new_state_dict[key] = value


state_dict = new_state_dict


# ============================================================
# 11. Load model weights
# ============================================================

msg = model.load_state_dict(
    state_dict,
    strict=False
)

print('Checkpoint loading result:')
print(msg)
print()


# ============================================================
# 12. Move model to GPU/CPU
# ============================================================

model = model.to(device)

model.eval()

print('Model loaded successfully.')
print()


# ============================================================
# 13. Collect all test images
# ============================================================

image_extensions = (
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff'
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


print('Total test images:', len(image_paths))
print()


# ============================================================
# 14. Prediction lists
# ============================================================

actual_labels = []

predicted_labels = []

results = []


# ============================================================
# 15. Test every image
# ============================================================

print('Starting prediction...')
print()


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

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert('RGB')


        # ----------------------------------------------------
        # Keep original image for visualization
        # ----------------------------------------------------

        original_image = image.copy()


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        input_tensor = transform(
            image
        )

        input_tensor = input_tensor.unsqueeze(
            0
        )

        input_tensor = input_tensor.to(
            device
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        predicted_index = (
            prediction.item()
        )

        confidence_value = (
            confidence.item()
        )


        predicted_class = idx_to_class[
            predicted_index
        ]


        # ----------------------------------------------------
        # Save labels
        # ----------------------------------------------------

        actual_labels.append(
            class_to_idx[actual_class]
        )

        predicted_labels.append(
            predicted_index
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            'image':
                filename,

            'image_path':
                image_path,

            'actual':
                actual_class,

            'predicted':
                predicted_class,

            'confidence':
                confidence_value,

            'correct':
                actual_class == predicted_class

        })


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            '[{}/{}] {}'
            .format(
                image_number,
                len(image_paths),
                filename
            )
        )

        print(
            '    Actual    : {}'
            .format(actual_class)
        )

        print(
            '    Predicted : {}'
            .format(predicted_class)
        )

        print(
            '    Confidence: {:.4f}'
            .format(confidence_value)
        )

        print()


        # ====================================================
        # Create visualization
        # ====================================================

        plt.figure(
            figsize=(8, 7)
        )

        plt.imshow(
            original_image
        )

        plt.axis('off')


        # ----------------------------------------------------
        # Determine title
        # ----------------------------------------------------

        if actual_class == predicted_class:

            title = (
                'Actual: {}\n'
                'Predicted: {} '
                '({:.2f}%)'
            ).format(
                actual_class,
                predicted_class,
                confidence_value * 100
            )

        else:

            title = (
                'Actual: {}\n'
                'Predicted: {} '
                '({:.2f}%)'
            ).format(
                actual_class,
                predicted_class,
                confidence_value * 100
            )


        plt.title(
            title,
            fontsize=14
        )


        # ----------------------------------------------------
        # Create unique output filename
        # ----------------------------------------------------

        safe_filename = (
            filename
            .replace(
                '.jpg',
                ''
            )
            .replace(
                '.jpeg',
                ''
            )
            .replace(
                '.png',
                ''
            )
        )


        output_filename = (
            '{}_actual_{}_pred_{}.png'
            .format(
                safe_filename,
                actual_class.replace(
                    ' ',
                    '_'
                ),
                predicted_class.replace(
                    ' ',
                    '_'
                )
            )
        )


        output_image_path = os.path.join(
            IMAGE_OUTPUT_DIR,
            output_filename
        )


        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        plt.savefig(
            output_image_path,
            bbox_inches='tight',
            dpi=200
        )

        plt.close()


    except Exception as e:

        print(
            'ERROR processing:',
            image_path
        )

        print(
            'Error:',
            str(e)
        )

        print()

        continue


# ============================================================
# 16. Save predictions CSV
# ============================================================

df = pd.DataFrame(
    results
)

df.to_csv(
    CSV_PATH,
    index=False
)


print('==========================================')
print('Prediction completed')
print('==========================================')
print()


# ============================================================
# 17. Calculate accuracy
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)


print(
    'Test Accuracy: {:.4f}'
    .format(accuracy)
)

print(
    'Test Accuracy: {:.2f}%'
    .format(accuracy * 100)
)

print()


# ============================================================
# 18. Classification report
# ============================================================

report = classification_report(
    actual_labels,
    predicted_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print('Classification Report')
print('==========================================')
print(report)


# ============================================================
# 19. Save classification report
# ============================================================

with open(
    REPORT_PATH,
    'w'
) as f:

    f.write(
        'RETFound Classification Report\n'
    )

    f.write(
        '==========================================\n\n'
    )

    f.write(
        'Test Directory:\n'
    )

    f.write(
        TEST_DIR + '\n\n'
    )

    f.write(
        'Checkpoint:\n'
    )

    f.write(
        CHECKPOINT_PATH + '\n\n'
    )

    f.write(
        'Number of classes: {}\n'.format(
            NUM_CLASSES
        )
    )

    f.write(
        'Number of images: {}\n'.format(
            len(results)
        )
    )

    f.write(
        'Accuracy: {:.4f}\n'.format(
            accuracy
        )
    )

    f.write(
        'Accuracy: {:.2f}%\n\n'.format(
            accuracy * 100
        )
    )

    f.write(
        report
    )


# ============================================================
# 20. Confusion matrix
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
    'RETFound - Frisen Grade Confusion Matrix'
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches='tight'
)


plt.show()

plt.close()


# ============================================================
# 21. Save summary
# ============================================================

total_images = len(results)

correct_predictions = sum(
    1
    for result in results
    if result['correct']
)

incorrect_predictions = (
    total_images -
    correct_predictions
)


with open(
    SUMMARY_PATH,
    'w'
) as f:

    f.write(
        'RETFound Test Summary\n'
    )

    f.write(
        '==========================================\n\n'
    )

    f.write(
        'Test directory: {}\n'.format(
            TEST_DIR
        )
    )

    f.write(
        'Checkpoint: {}\n'.format(
            CHECKPOINT_PATH
        )
    )

    f.write(
        'Number of classes: {}\n'.format(
            NUM_CLASSES
        )
    )

    f.write(
        'Classes: {}\n\n'.format(
            ', '.join(class_names)
        )
    )

    f.write(
        'Total images: {}\n'.format(
            total_images
        )
    )

    f.write(
        'Correct predictions: {}\n'.format(
            correct_predictions
        )
    )

    f.write(
        'Incorrect predictions: {}\n'.format(
            incorrect_predictions
        )
    )

    f.write(
        'Accuracy: {:.2f}%\n'.format(
            accuracy * 100
        )
    )


# ============================================================
# 22. Final output
# ============================================================

print()
print('==========================================')
print('ALL RESULTS SAVED')
print('==========================================')

print(
    'Predictions CSV:'
)

print(
    CSV_PATH
)

print()

print(
    'Individual prediction images:'
)

print(
    IMAGE_OUTPUT_DIR
)

print()

print(
    'Confusion matrix:'
)

print(
    CONFUSION_MATRIX_PATH
)

print()

print(
    'Classification report:'
)

print(
    REPORT_PATH
)

print()

print(
    'Summary:'
)

print(
    SUMMARY_PATH
)

print('==========================================')