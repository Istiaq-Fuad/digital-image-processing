import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Contrast Adjustment
# -----------------------------------
def adjust_contrast(img, alpha=1.0):
    img_float = img.astype(np.float32)
    result = alpha * img_float
    return np.clip(result, 0, 255).astype(np.uint8)


# -----------------------------------
# FFT and Spectrum
# -----------------------------------
def compute_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    return fshift, magnitude


# -----------------------------------
# Distance Matrix (Vectorized)
# -----------------------------------
def distance_matrix(shape):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    return D


# -----------------------------------
# Create Frequency Masks (Using Distance Matrix)
# -----------------------------------
def create_filter_mask(shape, filter_type="low", radius=30, r_out=60):
    D = distance_matrix(shape)

    if filter_type == "low":
        mask = D <= radius

    elif filter_type == "high":
        mask = D >= radius

    elif filter_type == "band":
        mask = (D >= radius) & (D <= r_out)

    return mask.astype(np.uint8)


# -----------------------------------
# Apply Frequency Filter
# -----------------------------------
def apply_filter(fshift, mask):
    filtered = fshift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))


# -----------------------------------
# Display Function
# -----------------------------------
def show_results(images, titles):
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------------
# MAIN
# -----------------------------------
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

low_contrast = adjust_contrast(img, alpha=0.5)
normal_contrast = img.copy()
high_contrast = adjust_contrast(img, alpha=1.8)

# Compute FFTs
f_low, mag_low = compute_fft(low_contrast)
f_norm, mag_norm = compute_fft(normal_contrast)
f_high, mag_high = compute_fft(high_contrast)

# Create filters using distance matrix
mask_low = create_filter_mask(img.shape, "low", radius=40)
mask_high = create_filter_mask(img.shape, "high", radius=40)
mask_band = create_filter_mask(img.shape, "band", radius=20, r_out=60)

# Apply filters
lpf_img = apply_filter(f_norm, mask_low)
hpf_img = apply_filter(f_norm, mask_high)
bpf_img = apply_filter(f_norm, mask_band)

# Show results
images = [
    low_contrast,
    mag_low,
    normal_contrast,
    mag_norm,
    high_contrast,
    mag_high,
    lpf_img,
    hpf_img,
    bpf_img,
]

titles = [
    "Low Contrast",
    "Low Contrast Spectrum",
    "Normal Contrast",
    "Normal Spectrum",
    "High Contrast",
    "High Contrast Spectrum",
    "Low Pass Filtered",
    "High Pass Filtered",
    "Band Pass Filtered",
]

show_results(images, titles)
