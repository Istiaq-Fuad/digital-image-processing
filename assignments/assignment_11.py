import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "images/deer.jpg"
S = 3
CLAHE_TILES = 8


def he_gray(gray):
    return cv.equalizeHist(gray)


def clahe_gray(gray, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def ahe_bilinear(gray, tile_grid=(8, 8)):
    return clahe_gray(gray, clip_limit=100.0, tile_grid=tile_grid)


def ahe_naive_no_interp(gray, tiles=(8, 8)):
    ty, tx = tiles
    h, w = gray.shape
    ys = np.linspace(0, h, ty + 1, dtype=int)
    xs = np.linspace(0, w, tx + 1, dtype=int)

    out = np.empty_like(gray)
    for i in range(ty):
        for j in range(tx):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            out[y0:y1, x0:x1] = cv.equalizeHist(gray[y0:y1, x0:x1])
    return out


def linear_contrast_brightness(gray, alpha=1.2, beta=10):
    return cv.convertScaleAbs(gray, alpha=alpha, beta=beta)


def gamma_correction(gray, gamma=1.0):
    gamma = max(1e-6, float(gamma))
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255.0).clip(0, 255).astype(np.uint8)
    return cv.LUT(gray, table)


def gaussian_blur(gray, k=5, sigma=1.0):
    k = k if k % 2 == 1 else k + 1
    return cv.GaussianBlur(gray, (k, k), sigma)


def median_blur(gray, k=5):
    k = k if k % 2 == 1 else k + 1
    return cv.medianBlur(gray, k)


def unsharp_mask(gray, amount=1.0, sigma=1.0):
    blur = cv.GaussianBlur(gray, (0, 0), sigma)
    return cv.addWeighted(gray, 1 + amount, blur, -amount, 0)


def op_original(tile):
    return tile


def op_linear_plus(tile):
    return linear_contrast_brightness(tile, 1.2, 10)


def op_linear_minus(tile):
    return linear_contrast_brightness(tile, 0.9, -10)


def op_gamma_07(tile):
    return gamma_correction(tile, 0.7)


def op_gamma_15(tile):
    return gamma_correction(tile, 1.5)


def op_gaussian_5(tile):
    return gaussian_blur(tile, 5, 1.0)


def op_median_5(tile):
    return median_blur(tile, 5)


def op_unsharp(tile):
    return unsharp_mask(tile, 1.0, 1.0)


def op_he(tile):
    return he_gray(tile)


def op_ahe_bilinear(tile):
    return ahe_bilinear(tile, (CLAHE_TILES, CLAHE_TILES))


def op_clahe2(tile):
    return clahe_gray(tile, 2.0, (CLAHE_TILES, CLAHE_TILES))


def op_clahe4(tile):
    return clahe_gray(tile, 4.0, (CLAHE_TILES, CLAHE_TILES))


OPS = [
    op_original,
    op_linear_plus,
    op_linear_minus,
    op_gamma_07,
    op_gamma_15,
    op_gaussian_5,
    op_median_5,
    op_unsharp,
    op_he,
    op_ahe_bilinear,
    op_clahe2,
    op_clahe4,
]


def make_mosaic(gray, s, ops):
    h, w = gray.shape
    ys = np.linspace(0, h, s + 1, dtype=int)
    xs = np.linspace(0, w, s + 1, dtype=int)
    out = np.zeros_like(gray)
    k = 0
    for i in range(s):
        for j in range(s):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            tile = gray[y0:y1, x0:x1]
            try:
                out[y0:y1, x0:x1] = ops[k % len(ops)](tile)
            except Exception:
                out[y0:y1, x0:x1] = tile
            k += 1
    return out


img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

original = img
he_result = he_gray(img)
ahe_bilinear_result = ahe_bilinear(img, (CLAHE_TILES, CLAHE_TILES))
clahe_2_result = clahe_gray(img, 2.0, (CLAHE_TILES, CLAHE_TILES))
clahe_4_result = clahe_gray(img, 4.0, (CLAHE_TILES, CLAHE_TILES))
ahe_no_interp_result = ahe_naive_no_interp(img, (CLAHE_TILES, CLAHE_TILES))
mosaic = make_mosaic(img, S, OPS)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Histogram Equalization Techniques Comparison", fontsize=16)

axes[0, 0].imshow(original, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(he_result, cmap="gray")
axes[0, 1].set_title("Histogram Equalization")
axes[0, 1].axis("off")

axes[0, 2].imshow(ahe_bilinear_result, cmap="gray")
axes[0, 2].set_title("AHE (Bilinear)")
axes[0, 2].axis("off")

axes[0, 3].imshow(clahe_2_result, cmap="gray")
axes[0, 3].set_title("CLAHE (Clip=2.0)")
axes[0, 3].axis("off")

axes[1, 0].imshow(clahe_4_result, cmap="gray")
axes[1, 0].set_title("CLAHE (Clip=4.0)")
axes[1, 0].axis("off")

axes[1, 1].imshow(ahe_no_interp_result, cmap="gray")
axes[1, 1].set_title("AHE (No Interpolation)")
axes[1, 1].axis("off")

axes[1, 2].imshow(mosaic, cmap="gray")
axes[1, 2].set_title("Mosaic (All Techniques)")
axes[1, 2].axis("off")

axes[1, 3].remove()

plt.tight_layout()

plt.savefig("images/output/assignment_11_comparison.png")

plt.show()
