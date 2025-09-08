import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(image):
    hist = np.zeros(256, dtype=np.int32)
    for pixel in image.flatten():
        hist[pixel] += 1
    return hist


def brightness_shift(image, b):
    shifted = image.astype(np.int16) + b
    shifted = np.clip(shifted, 0, 255)

    return shifted.astype(np.uint8)


def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(1 + image.astype(np.float64))
    return np.clip(log_image, 0, 255).astype(np.uint8)


image = cv2.imread("my_image.jpeg", 0)

linear_result = brightness_shift(image, 80)
nonlinear_result = log_transform(image)

plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.bar(range(256), compute_histogram(image), color="black")
plt.title("Original Histogram")

plt.subplot(3, 2, 3)
plt.imshow(linear_result, cmap="gray")
plt.title("Linear: Brightness Shift")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.bar(range(256), compute_histogram(linear_result), color="black")
plt.title("Linear Histogram")

plt.subplot(3, 2, 5)
plt.imshow(nonlinear_result, cmap="gray")
plt.title("Nonlinear: Log Transform")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.bar(range(256), compute_histogram(nonlinear_result), color="black")
plt.title("Nonlinear Histogram")

plt.tight_layout()
plt.show()
