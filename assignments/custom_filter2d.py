import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def conv2d(image, kernel, mode="same"):
    image = np.array(image, dtype=np.float32)
    kernel = np.array(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    ih, iw = image.shape

    kernel = np.flipud(np.fliplr(kernel))

    if mode == "same":
        pad_h = kh // 2
        pad_w = kw // 2
        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )
    elif mode == "valid":
        padded = image
    else:
        raise ValueError("Mode must be 'same' or 'valid'")

    ph, pw = padded.shape
    oh = ph - kh + 1
    ow = pw - kw + 1
    output = np.zeros((oh, ow), dtype=np.float32)

    for i in range(oh):
        for j in range(ow):
            region = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


avg_kernel = np.ones((3, 3), dtype=np.float32) / 9
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
custom1 = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], dtype=np.float32)
custom2 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32)
custom3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
custom4 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

kernels = {
    "average": avg_kernel,
    "sobel_x": sobel_x,
    "sobel_y": sobel_y,
    "prewitt_x": prewitt_x,
    "prewitt_y": prewitt_y,
    "scharr_x": scharr_x,
    "scharr_y": scharr_y,
    "laplace": laplace_kernel,
    "custom1": custom1,
    "custom2": custom2,
    "custom3": custom3,
    "custom4": custom4,
}


image_path = "images/deer.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

scale_percent = 30
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for idx, (name, k) in enumerate(kernels.items()):
    filtered = conv2d(img_resized, k, mode="same")
    filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    axes[idx].imshow(filtered_norm, cmap="gray")
    axes[idx].set_title(name)
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig("images/output/filtered_kernels_same_mode.png", dpi=300)
plt.show()
print("All filtered images saved in 'images/output/filtered_kernels_same_mode.png'.")
