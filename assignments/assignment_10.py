import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def custom_erode(img, kernel):
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)))
    eroded_img = np.zeros_like(img)

    for i in range(eroded_img.shape[0]):
        for j in range(eroded_img.shape[1]):
            region = padded_img[i : i + k_height, j : j + k_width]
            if np.all(region[kernel == 1] == 1):
                eroded_img[i, j] = 1
            else:
                eroded_img[i, j] = 0

    return eroded_img


def custom_dilate(img, kernel):
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)))
    dilated_img = np.zeros_like(img)

    for i in range(dilated_img.shape[0]):
        for j in range(dilated_img.shape[1]):
            region = padded_img[i : i + k_height, j : j + k_width]
            if np.any(region[kernel == 1] == 1):
                dilated_img[i, j] = 1
            else:
                dilated_img[i, j] = 0

    return dilated_img


def custom_open(img, kernel):
    return custom_dilate(custom_erode(img, kernel), kernel)


def custom_close(img, kernel):
    return custom_erode(custom_dilate(img, kernel), kernel)


def custom_tophat(img, kernel):
    return cv2.subtract(img, custom_open(img, kernel))


def custom_blackhat(img, kernel):
    return cv2.subtract(custom_close(img, kernel), img)


rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
diamond = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

structuring_elements = {
    "rect": rect,
    "ellipse": ellipse,
    "cross": cross,
    "diamond": diamond,
}


img = cv2.imread("images/deer.jpg", 0)
img_small = cv2.resize(img, (256, 256))
_, img_bin = cv2.threshold(img_small, 127, 255, cv2.THRESH_BINARY)

operations = {
    "Erosion": (cv2.erode, custom_erode),
    "Dilation": (cv2.dilate, custom_dilate),
    "Opening": (lambda im, k: cv2.morphologyEx(im, cv2.MORPH_OPEN, k), custom_open),
    "Closing": (
        lambda im, k: cv2.morphologyEx(im, cv2.MORPH_CLOSE, k),
        custom_close,
    ),
    "Top-Hat": (
        lambda im, k: cv2.morphologyEx(im, cv2.MORPH_TOPHAT, k),
        custom_tophat,
    ),
    "Black-Hat": (
        lambda im, k: cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, k),
        custom_blackhat,
    ),
}

for se_name, kernel in structuring_elements.items():
    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(
        f"Morphological Operations with {se_name.capitalize()} Structuring Element",
        fontsize=14,
    )

    for idx, (op_name, (opencv_fn, custom_fn)) in enumerate(operations.items()):

        opencv_res = opencv_fn(img_bin, kernel)
        axs[0, idx].imshow(opencv_res, cmap="gray")
        axs[0, idx].set_title(f"{op_name}\n(OpenCV)", fontsize=8)
        axs[0, idx].axis("off")

        custom_res = custom_fn(img_bin, kernel)
        axs[1, idx].imshow(custom_res, cmap="gray")
        axs[1, idx].set_title(f"{op_name}\n(Custom)", fontsize=8)
        axs[1, idx].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.savefig(f"images/output/{se_name}_all.png", dpi=150)
    plt.close()
