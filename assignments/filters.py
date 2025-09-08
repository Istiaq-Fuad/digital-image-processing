import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)


avg_kernel = np.ones((3, 3), dtype=np.float32) / 9


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


avg_filtered = cv2.filter2D(img, -1, avg_kernel)
sobel_x_filtered = cv2.filter2D(img, -1, sobel_x)
sobel_y_filtered = cv2.filter2D(img, -1, sobel_y)
prewitt_x_filtered = cv2.filter2D(img, -1, prewitt_x)
prewitt_y_filtered = cv2.filter2D(img, -1, prewitt_y)
laplace_filtered = cv2.filter2D(img, -1, laplace_kernel)


sobel_magnitude = np.sqrt(sobel_x_filtered**2 + sobel_y_filtered**2)
prewitt_magnitude = np.sqrt(prewitt_x_filtered**2 + prewitt_y_filtered**2)


plt.figure(figsize=(20, 8))

plt.subplot(2, 5, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 5, 2)
plt.imshow(avg_filtered, cmap="gray")
plt.title("Average Filter")
plt.axis("off")

plt.subplot(2, 5, 3)
plt.imshow(sobel_x_filtered, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 5, 4)
plt.imshow(sobel_y_filtered, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 5, 5)
plt.imshow(sobel_magnitude, cmap="gray")
plt.title("Sobel Magnitude")
plt.axis("off")

plt.subplot(2, 5, 6)
plt.imshow(prewitt_x_filtered, cmap="gray")
plt.title("Prewitt X")
plt.axis("off")

plt.subplot(2, 5, 7)
plt.imshow(prewitt_y_filtered, cmap="gray")
plt.title("Prewitt Y")
plt.axis("off")

plt.subplot(2, 5, 8)
plt.imshow(prewitt_magnitude, cmap="gray")
plt.title("Prewitt Magnitude")
plt.axis("off")

plt.subplot(2, 5, 9)
plt.imshow(laplace_filtered, cmap="gray")
plt.title("Laplace Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
