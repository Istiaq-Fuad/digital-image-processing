import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# -----------------------------------------
# Load Image (Grayscale)
# -----------------------------------------
img = cv2.imread("../images/flower.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# -----------------------------------------
# 1️⃣ DFT using NumPy
# -----------------------------------------
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
dft_magnitude = np.log(1 + np.abs(dft_shift))

# -----------------------------------------
# 2️⃣ DCT using OpenCV
# -----------------------------------------
img_float = np.float32(img)   # Required for cv2.dct()
dct_transformed = cv2.dct(img_float)
dct_magnitude = np.log(1 + np.abs(dct_transformed))

# -----------------------------------------
# 3️⃣ DWT (Discrete Wavelet Transform)
# -----------------------------------------
coeffs2 = pywt.dwt2(img, 'haar')
LL, (LH, HL, HH) = coeffs2

LH_vis = np.abs(LH)
HL_vis = np.abs(HL)
HH_vis = np.abs(HH)

# -----------------------------------------
# Visualization
# -----------------------------------------
plt.figure(figsize=(16, 12))

plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(dft_magnitude, cmap='gray')
plt.title("DFT Magnitude Spectrum")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(dct_magnitude, cmap='gray')
plt.title("DCT Coefficients (cv2)")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(LL, cmap='gray')
plt.title("DWT - LL (Approximation)")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(LH_vis, cmap='gray')
plt.title("DWT - LH (Horizontal Details)")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(HL_vis, cmap='gray')
plt.title("DWT - HL (Vertical Details)")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(HH_vis, cmap='gray')
plt.title("DWT - HH (Diagonal Details)")
plt.axis("off")

plt.tight_layout()
plt.show()