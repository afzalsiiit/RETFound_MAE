import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 1. Add RETFound_MAE to Python path
# ============================================================

sys.path.insert(0, '/content/RETFound_MAE')

import models_mae


# ============================================================
# 2. ImageNet normalization
# ============================================================

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


# ============================================================
# 3. Display image
# ============================================================

def show_image(image, title=''):
    """
    image: [H, W, 3]
    """

    assert image.shape[2] == 3

    image = torch.clip(
        (image * imagenet_std + imagenet_mean) * 255,
        0,
        255
    ).int()

    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')


# ============================================================
# 4. Load RETFound MAE model
# ============================================================

def prepare_model(
    chkpt_dir,
    arch='mae_vit_large_patch16'
):

    # Build model
    model = getattr(models_mae, arch)()

    # Load checkpoint
    checkpoint = torch.load(
        chkpt_dir,
        map_location='cpu'
    )

    msg = model.load_state_dict(
        checkpoint['model'],
        strict=False
    )

    print(msg)

    return model


# ============================================================
# 5. Run MAE reconstruction
# ============================================================

def run_one_image(
    img,
    model,
    output_path
):

    # --------------------------------------------------------
    # Convert image to tensor
    # --------------------------------------------------------

    x = torch.tensor(img)

    # Add batch dimension
    x = x.unsqueeze(dim=0)

    # NHWC -> NCHW
    x = torch.einsum(
        'nhwc->nchw',
        x
    )

    # --------------------------------------------------------
    # Run MAE
    # --------------------------------------------------------

    with torch.no_grad():

        loss, y, mask = model(
            x.float(),
            mask_ratio=0.75
        )

    # --------------------------------------------------------
    # Reconstruct image
    # --------------------------------------------------------

    y = model.unpatchify(y)

    # NCHW -> NHWC
    y = torch.einsum(
        'nchw->nhwc',
        y
    ).detach().cpu()

    # --------------------------------------------------------
    # Create visualization mask
    # --------------------------------------------------------

    mask = mask.detach()

    mask = mask.unsqueeze(-1).repeat(
        1,
        1,
        model.patch_embed.patch_size[0] ** 2 * 3
    )

    mask = model.unpatchify(mask)

    # NCHW -> NHWC
    mask = torch.einsum(
        'nchw->nhwc',
        mask
    ).detach().cpu()

    # Original image
    x = torch.einsum(
        'nchw->nhwc',
        x
    )

    # --------------------------------------------------------
    # Masked image
    # --------------------------------------------------------

    im_masked = x * (1 - mask)

    # --------------------------------------------------------
    # Reconstruction + visible patches
    # --------------------------------------------------------

    im_paste = x * (1 - mask) + y * mask

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(figsize=(24, 6))

    # Original
    plt.subplot(1, 4, 1)
    show_image(
        x[0],
        "Original"
    )

    # Masked
    plt.subplot(1, 4, 2)
    show_image(
        im_masked[0],
        "Masked"
    )

    # Reconstruction
    plt.subplot(1, 4, 3)
    show_image(
        y[0],
        "Reconstruction"
    )

    # Reconstruction + visible
    plt.subplot(1, 4, 4)
    show_image(
        im_paste[0],
        "Reconstruction + Visible"
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    plt.savefig(
        output_path,
        bbox_inches='tight',
        dpi=300
    )

    print()
    print("==========================================")
    print("MAE reconstruction completed")
    print("==========================================")
    print("Saved result:")
    print(output_path)
    print("==========================================")

    # Display
    plt.show()

    # Close figure
    plt.close()


# ============================================================
# 6. Load input image
# ============================================================

img_path = (
    '/content/SULYAP-Image-Classification-1/'
    'test/Frisen Grade 0/'
    '342_jpg.rf.50af5b8fda39aee01dbd07bdefb5ff32.jpg'
)

print("Loading image:")
print(img_path)

img = Image.open(
    img_path
).convert('RGB')


# ============================================================
# 7. Resize image
# ============================================================

img = img.resize(
    (224, 224)
)


# ============================================================
# 8. Convert image to NumPy
# ============================================================

img = np.array(img) / 255.0

assert img.shape == (224, 224, 3)


# ============================================================
# 9. Normalize using ImageNet mean/std
# ============================================================

img = img - imagenet_mean
img = img / imagenet_std


# ============================================================
# 10. Show original image
# ============================================================

plt.figure(figsize=(5, 5))

show_image(
    torch.tensor(img),
    "Input Image"
)

plt.show()
plt.close()


# ============================================================
# 11. Load RETFound checkpoint
# ============================================================

chkpt_dir = (
    '/content/drive/MyDrive/REFT/checkpoint-best.pth'
)

print()
print("Loading RETFound checkpoint:")
print(chkpt_dir)

model_mae = prepare_model(
    chkpt_dir,
    'mae_vit_large_patch16'
)

print()
print("Model loaded successfully.")


# ============================================================
# 12. Reproducible masking
# ============================================================

torch.manual_seed(2)


# ============================================================
# 13. Create output directory
# ============================================================

output_dir = '/content/mae_results'

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 14. Output filename
# ============================================================

output_path = os.path.join(
    output_dir,
    '342_mae_reconstruction.png'
)


# ============================================================
# 15. Run reconstruction and save result
# ============================================================

print()
print("MAE with pixel reconstruction:")

run_one_image(
    img,
    model_mae,
    output_path
)