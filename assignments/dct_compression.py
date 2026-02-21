import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../images/flower.jpg", 0)
img = cv2.resize(img, (256, 256))
img = np.float32(img)

Q = np.ones((8, 8)) * 20  # Simple quantization matrix

compressed = np.zeros_like(img)

for i in range(0, 256, 8):
    for j in range(0, 256, 8):

        block = img[i : i + 8, j : j + 8]

        dct_block = cv2.dct(block)
        quantized = np.round(dct_block / Q)
        dequantized = quantized * Q

        reconstructed = cv2.idct(dequantized)
        compressed[i : i + 8, j : j + 8] = reconstructed

compressed = np.uint8(np.clip(compressed, 0, 255))

plt.subplot(1, 2, 1)
plt.imshow(img.astype(np.uint8), cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(compressed, cmap="gray")
plt.title("DCT Compressed")

plt.show()
