import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utility Functions
# -----------------------------

def adjust_contrast(img, mode="normal"):
    img = img.astype(np.float32)

    if mode == "low":
        img = img * 0.5 + 60      # reduce contrast
    elif mode == "high":
        img = img * 1.5 - 40      # increase contrast

    return np.clip(img, 0, 255).astype(np.uint8)


def fft2_image(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def ifft2_image(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))


def distance_matrix(shape):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((x - ccol)**2 + (y - crow)**2)
    return D


# -----------------------------
# Ideal Filters
# -----------------------------

def ideal_lpf(shape, D0):
    D = distance_matrix(shape)
    H = np.zeros(shape)
    H[D <= D0] = 1
    return H


def ideal_hpf(shape, D0):
    return 1 - ideal_lpf(shape, D0)


def ideal_bpf(shape, D0, W):
    D = distance_matrix(shape)
    H = np.zeros(shape)
    H[(D >= (D0 - W/2)) & (D <= (D0 + W/2))] = 1
    return H


# -----------------------------
# Butterworth Filters
# -----------------------------

def butterworth_lpf(shape, D0, n):
    D = distance_matrix(shape)
    return 1 / (1 + (D / D0)**(2*n))


def butterworth_hpf(shape, D0, n):
    return 1 - butterworth_lpf(shape, D0, n)


def butterworth_bpf(shape, D0, W, n):
    D = distance_matrix(shape)
    return 1 / (1 + ((D*W)/(D**2 - D0**2 + 1e-5))**(2*n))


# -----------------------------
# Gaussian Filters
# -----------------------------

def gaussian_lpf(shape, D0):
    D = distance_matrix(shape)
    return np.exp(-(D**2) / (2*(D0**2)))


def gaussian_hpf(shape, D0):
    return 1 - gaussian_lpf(shape, D0)


def gaussian_bpf(shape, D0, W):
    D = distance_matrix(shape)
    return np.exp(-((D**2 - D0**2)**2) / (D**2 * W**2 + 1e-5))


# -----------------------------
# Apply Filter in Frequency Domain
# -----------------------------

def apply_filter(img, H):
    F = fft2_image(img)
    G = F * H
    result = ifft2_image(G)
    return result


# -----------------------------
# Main Program
# -----------------------------

# Load grayscale image
img = cv2.imread("image.jpg", 0)

# Contrast versions
img_low = adjust_contrast(img, "low")
img_normal = adjust_contrast(img, "normal")
img_high = adjust_contrast(img, "high")

contrast_images = {
    "Low Contrast": img_low,
    "Normal Contrast": img_normal,
    "High Contrast": img_high
}

D0 = 40
W = 20
orders = [1, 2, 5]   # Different n values

for contrast_name, image in contrast_images.items():

    plt.figure(figsize=(18, 12))
    plt.suptitle(f"{contrast_name}", fontsize=16)

    # Ideal Filters
    ideal_lp = apply_filter(image, ideal_lpf(image.shape, D0))
    ideal_hp = apply_filter(image, ideal_hpf(image.shape, D0))
    ideal_bp = apply_filter(image, ideal_bpf(image.shape, D0, W))

    # Gaussian Filters
    gauss_lp = apply_filter(image, gaussian_lpf(image.shape, D0))
    gauss_hp = apply_filter(image, gaussian_hpf(image.shape, D0))
    gauss_bp = apply_filter(image, gaussian_bpf(image.shape, D0, W))

    # Butterworth (for n=2 default comparison)
    butter_lp = apply_filter(image, butterworth_lpf(image.shape, D0, 2))
    butter_hp = apply_filter(image, butterworth_hpf(image.shape, D0, 2))
    butter_bp = apply_filter(image, butterworth_bpf(image.shape, D0, W, 2))

    results = [
        image,
        ideal_lp, ideal_hp, ideal_bp,
        butter_lp, butter_hp, butter_bp,
        gauss_lp, gauss_hp, gauss_bp
    ]

    titles = [
        "Original",
        "Ideal LPF", "Ideal HPF", "Ideal BPF",
        "Butterworth LPF", "Butterworth HPF", "Butterworth BPF",
        "Gaussian LPF", "Gaussian HPF", "Gaussian BPF"
    ]

    for i in range(len(results)):
        plt.subplot(3, 4, i+1)
        plt.imshow(results[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Effect of Different n (Butterworth)
# -----------------------------

plt.figure(figsize=(15, 5))
plt.suptitle("Effect of Different n in Butterworth LPF")

for i, n in enumerate(orders):
    butter_lp = apply_filter(img, butterworth_lpf(img.shape, D0, n))
    plt.subplot(1, 3, i+1)
    plt.imshow(butter_lp, cmap="gray")
    plt.title(f"n = {n}")
    plt.axis("off")

plt.show()