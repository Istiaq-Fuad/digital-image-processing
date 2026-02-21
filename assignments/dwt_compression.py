import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("input.jpg", 0)
img = cv2.resize(img, (256, 256))

# Apply DWT
coeffs = pywt.dwt2(img, "haar")
LL, (LH, HL, HH) = coeffs

# Threshold small coefficients
threshold = 20
LH[np.abs(LH) < threshold] = 0
HL[np.abs(HL) < threshold] = 0
HH[np.abs(HH) < threshold] = 0

# Reconstruct
compressed_dwt = pywt.idwt2((LL, (LH, HL, HH)), "haar")
compressed_dwt = np.uint8(np.clip(compressed_dwt, 0, 255))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(compressed_dwt, cmap="gray")
plt.title("DWT Compressed")

plt.show()
