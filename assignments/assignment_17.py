import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import pywt


def apply_dft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum


def apply_dct(image):
    dct_rows = scipy.fftpack.dct(image, axis=0, norm="ortho")
    dct_cols = scipy.fftpack.dct(dct_rows, axis=1, norm="ortho")
    magnitude_spectrum = np.log(np.abs(dct_cols) + 1)
    return magnitude_spectrum


def apply_dwt(image):
    coeffs = pywt.dwt2(image, "haar")
    LL, (LH, HL, HH) = coeffs

    def normalize(band):
        band = np.log(np.abs(band) + 1)
        return cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)

    top_row = np.hstack((normalize(LL), normalize(HL)))
    bot_row = np.hstack((normalize(LH), normalize(HH)))
    dwt_viz = np.vstack((top_row, bot_row))

    return dwt_viz


img = cv2.imread("images/sunflower.png", 0)

if img is None:

    img = np.zeros((256, 256), dtype=np.uint8)
    img[::32, ::32] = 255
    img[1::32, 1::32] = 255

img_float = np.float32(img)


dft_result = apply_dft(img_float)
dct_result = apply_dct(img_float)
dwt_result = apply_dwt(img_float)


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
plt.suptitle("Comparison of Image Transforms: DFT, DCT, and DWT", fontsize=16)


axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original Grayscale Image")
axes[0, 0].axis("off")


axes[0, 1].imshow(dft_result, cmap="inferno")
axes[0, 1].set_title("DFT Magnitude Spectrum\n(Centered Low Frequencies)")
axes[0, 1].axis("off")


axes[1, 0].imshow(dct_result, cmap="inferno")
axes[1, 0].set_title("DCT Coefficients\n(Energy Compacted in Top-Left)")
axes[1, 0].axis("off")


axes[1, 1].imshow(dwt_result, cmap="gray")
axes[1, 1].set_title("DWT (Haar) Decomposition\n(LL, HL, LH, HH)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("images/output/transforms_comparison.png")
print("Comparison figure saved as 'transforms_comparison.png'")
