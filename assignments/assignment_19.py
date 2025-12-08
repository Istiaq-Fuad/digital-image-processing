import cv2
import numpy as np
import os
import scipy.fftpack
import pywt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

IMAGE_NAMES = [
    "berry.png",
    "birds.jpg",
    "deer.jpg",
    "flower.jpg",
    "human.jpg",
    "sunflower.png",
    "bull.jpg",
    "light.jpg",
    "pen.jpg",
    "ship.jpg",
]

BASE_DIR = "./images/"


def resize_image(image, width=256):
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_h = int(width * aspect_ratio)
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def calculate_metrics(original, compressed):
    if original.shape != compressed.shape:
        h, w, _ = original.shape
        compressed = cv2.resize(compressed, (w, h))

    orig_f = original.astype(np.float64)
    comp_f = compressed.astype(np.float64)

    mse_val = mse(orig_f, comp_f)

    if mse_val == 0:
        psnr_val = float("inf")
    else:
        psnr_val = psnr(orig_f, comp_f, data_range=255)

    ssim_val = ssim(orig_f, comp_f, data_range=255, channel_axis=2)

    return mse_val, psnr_val, ssim_val


def compress_rle(image):
    return image.copy()


def compress_dct(image, quantization_factor=20):
    h, w, c = image.shape
    compressed = np.zeros_like(image, dtype=np.float64)

    Q = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    ) * (quantization_factor / 10.0)

    for k in range(c):
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        channel = np.pad(image[:, :, k], ((0, h_pad), (0, w_pad)), "edge").astype(float)

        restored_channel = np.zeros_like(channel)

        for i in range(0, channel.shape[0], 8):
            for j in range(0, channel.shape[1], 8):
                block = channel[i : i + 8, j : j + 8] - 128
                dct_block = scipy.fftpack.dct(
                    scipy.fftpack.dct(block.T, norm="ortho").T, norm="ortho"
                )
                dct_quant = np.round(dct_block / Q)
                dct_dequant = dct_quant * Q
                idct_block = scipy.fftpack.idct(
                    scipy.fftpack.idct(dct_dequant.T, norm="ortho").T, norm="ortho"
                )
                restored_channel[i : i + 8, j : j + 8] = idct_block + 128

        compressed[:, :, k] = restored_channel[:h, :w]

    return np.clip(compressed, 0, 255).astype(np.uint8)


def compress_dwt(image, threshold=30):
    compressed = np.zeros_like(image)
    for k in range(3):
        coeffs = pywt.wavedec2(image[:, :, k], "haar", level=2)

        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = tuple(
                map(lambda x: pywt.threshold(x, threshold, mode="soft"), coeffs[i])
            )
        rec = pywt.waverec2(coeffs_thresh, "haar")

        h, w, _ = image.shape
        compressed[:, :, k] = np.clip(rec[:h, :w], 0, 255)

    return compressed.astype(np.uint8)


def compress_palette(image, k=256):
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(image.shape)


def compress_fixed_bits(image, r_bits, g_bits, b_bits):
    out = np.zeros_like(image)

    scale_r = 255 / (2**r_bits - 1)
    out[:, :, 0] = np.round(np.round(image[:, :, 0] / scale_r) * scale_r)

    scale_g = 255 / (2**g_bits - 1)
    out[:, :, 1] = np.round(np.round(image[:, :, 1] / scale_g) * scale_g)

    scale_b = 255 / (2**b_bits - 1)
    out[:, :, 2] = np.round(np.round(image[:, :, 2] / scale_b) * scale_b)
    return np.clip(out, 0, 255).astype(np.uint8)


def compress_general_depth(image, total_bits):
    bits_per_channel = total_bits // 3
    if bits_per_channel < 1:
        bits_per_channel = 1

    scale = 255 / (2**bits_per_channel - 1)
    out = np.round(np.round(image / scale) * scale)
    return np.clip(out, 0, 255).astype(np.uint8)


print(f"{'Image':<10} | {'Technique':<20} | {'MSE':<8} | {'PSNR':<8} | {'SSIM':<8}")
print("-" * 70)

for name in IMAGE_NAMES:
    path = os.path.join(BASE_DIR, name)
    img = cv2.imread(path)

    if img is None:
        print(f"Error: Could not load {path}. Checking next...")
        continue

    img = resize_image(img, width=256)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    techniques = {
        "RLE (Lossless)": compress_rle(img),
        "DCT (Q=20)": compress_dct(img, 20),
        "DWT (T=30)": compress_dwt(img, 30),
        "8-bit (Palette)": compress_palette(img, 256),
        "8-bit (3R-3G-2B)": compress_fixed_bits(img, 3, 3, 2),
        "8-bit (General)": compress_general_depth(img, 8),
        "5-bit (General)": compress_general_depth(img, 5),
        "4-bit (General)": compress_general_depth(img, 4),
        "2-bit (General)": compress_general_depth(img, 2),
        "1-bit (General)": compress_general_depth(img, 1),
    }

    for tech_name, compressed_img in techniques.items():
        mse_val, psnr_val, ssim_val = calculate_metrics(img, compressed_img)
        print(
            f"{name:<10} | {tech_name:<20} | {mse_val:8.2f} | {psnr_val:8.2f} | {ssim_val:8.4f}"
        )
    print("-" * 70)
