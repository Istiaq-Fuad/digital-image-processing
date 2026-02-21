import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_images(images, titles):
    plt.figure(figsize=(20, 20))
    n = len(images)
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, n, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")

        plt.subplot(2, n, idx + 1 + n)
        plt.hist(img.ravel(), bins=256, range=(0, 256), color="black")
        plt.title(f"Histogram - {title}")
    plt.tight_layout()
    # plt.savefig("images/output/histogram_matching_figure.png")
    plt.show()


def histogram_matching_cdf(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    src_pdf = src_hist / np.sum(src_hist)
    ref_pdf = ref_hist / np.sum(ref_hist)

    src_cdf = np.cumsum(src_pdf)
    ref_cdf = np.cumsum(ref_pdf)

    mapping = np.zeros(256, dtype=np.uint8)

    for src_intensity in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_intensity])
        mapping[src_intensity] = np.argmin(diff)

    matched = mapping[source]

    return matched


def histogram_matching_interpolation(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    src_pdf = src_hist / np.sum(src_hist)
    ref_pdf = ref_hist / np.sum(ref_hist)

    src_cdf = np.cumsum(src_pdf)
    ref_cdf = np.cumsum(ref_pdf)

    src_values = np.arange(256)
    ref_values = np.arange(256)

    interp_values = np.interp(src_cdf, ref_cdf, ref_values)

    matched = interp_values[source].astype(np.uint8)

    return matched


def adjust_contrast(img, level="normal"):
    img = img.astype(np.float32)
    if level == "low":
        img = img * 0.5 + 64
    elif level == "high":
        img = (img - 128) * 2 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def hist_corr(im1, im2):
    h1, _ = np.histogram(im1.flatten(), 256, [0, 256])
    h2, _ = np.histogram(im2.flatten(), 256, [0, 256])
    return np.corrcoef(h1, h2)[0, 1]


image = cv2.imread("../images/sunflower.png", cv2.IMREAD_GRAYSCALE)

src_low = adjust_contrast(image, "low")
ref_high = adjust_contrast(image, "high")

res_cdf = histogram_matching_cdf(src_low, ref_high)
res_spec = histogram_matching_interpolation(src_low, ref_high)

display_images(
    [src_low, ref_high, res_cdf, res_spec],
    ["Source (Low)", "Reference (High)", "Matched (CDF)", "Matched (Spec)"],
)

print(f"Histogram Correlation - CDF:  {hist_corr(ref_high, res_cdf):.4f}")
print(f"Histogram Correlation - Spec: {hist_corr(ref_high, res_spec):.4f}")
