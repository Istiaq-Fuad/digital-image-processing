import numpy as np
import cv2
import matplotlib.pyplot as plt


def custom_spatial_filter(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(
        image, ((pad_height, pad_height), (pad_width, pad_width)), mode="reflect"
    )

    output_image = np.zeros_like(image, dtype=image.dtype)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i : i + kernel_height, j : j + kernel_width]

            result = np.sum(region.astype(np.float64) * kernel.astype(np.float64))
            output_image[i, j] = (
                np.clip(result, 0, 255) if image.dtype == np.uint8 else result
            )

    return output_image


# test_image = np.array([
#     [1, 2, 3, 4, 5],
#     [6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15],
#     [16, 17, 18, 19, 20],
#     [21, 22, 23, 24, 25],
# ], dtype=np.float32)

# kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
# kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]], dtype=np.float32)
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)


image = cv2.imread("test1.jpg", cv2.IMREAD_GRAYSCALE)

image_float = image.astype(np.float32)

cv_output = cv2.filter2D(image_float, -1, kernel)
custom_output = custom_spatial_filter(image_float, kernel)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("OpenCV Output")
plt.imshow(cv_output, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Custom Output")
plt.imshow(custom_output, cmap="gray")

plt.tight_layout()
plt.show()
