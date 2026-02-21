import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utility: Show images
# -----------------------------
def show_images(images, titles, cmap="gray"):
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, (len(images) + 1) // 2, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Split image into s × s grids
# -----------------------------
def split_into_grids(img, s):
    h, w = img.shape
    grid_h = h // s
    grid_w = w // s

    grids = []
    for i in range(s):
        row = []
        for j in range(s):
            grid = img[i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
            row.append(grid)
        grids.append(row)

    return grids


# -----------------------------
# Reconstruct image from grids
# -----------------------------
def reconstruct_from_grids(grids):
    rows = [np.hstack(row) for row in grids]
    return np.vstack(rows)


# -----------------------------
# Linear Operations
# -----------------------------
def brightness_shift(img, beta=50):
    img_int = img.astype(np.int32)
    shifted = img_int + beta
    shifted = np.clip(shifted, 0, 255)
    return shifted.astype(np.uint8)


def contrast_stretch(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    return np.uint8(stretched)


# -----------------------------
# Non-Linear Operations
# -----------------------------
def gamma_correction(img, gamma=0.5):
    img_norm = img / 255.0
    gamma_img = np.power(img_norm, gamma)
    return np.uint8(gamma_img * 255)


def log_transform(img, c=1):
    img_float = img.astype(np.float32)
    log_img = c * np.log(1 + img_float)
    log_img = (log_img / np.max(log_img)) * 255
    return np.uint8(log_img)


# -----------------------------
# Main Processing
# -----------------------------
def process_image(image_path, s=2):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    grids = split_into_grids(img, s)

    # Apply different operations to different grids
    grids[0][0] = brightness_shift(grids[0][0], beta=50)  # Linear
    grids[0][1] = contrast_stretch(grids[0][1])  # Linear
    grids[1][0] = gamma_correction(grids[1][0], gamma=0.5)  # Non-linear
    grids[1][1] = log_transform(grids[1][1], c=1)  # Non-linear

    reconstructed = reconstruct_from_grids(grids)

    # Histogram Equalization
    he = cv2.equalizeHist(img)

    # Adaptive Histogram Equalization (AHE via CLAHE with high clip)
    ahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8)).apply(img)

    # CLAHE with different clip limits
    clahe_low = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    clahe_high = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)).apply(img)

    # Display results
    images = [img, reconstructed, he, ahe, clahe_low, clahe_high]
    titles = [
        "Original",
        "Grid-wise Linear & Nonlinear",
        "Histogram Equalization",
        "AHE (High Clip)",
        "CLAHE (Clip=2)",
        "CLAHE (Clip=10)",
    ]

    show_images(images, titles)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    process_image("../images/birds.jpg", s=2)
