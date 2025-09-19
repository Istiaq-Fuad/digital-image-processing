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
    # save image
    plt.savefig("images/output/histogram_matching_figure.png")
    plt.show()


def histogram_matching_cdf(src, ref):
    hist_src, _ = np.histogram(src.flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(ref.flatten(), 256, [0, 256])

    cdf_src = hist_src.cumsum() / hist_src.sum()
    cdf_ref = hist_ref.cumsum() / hist_ref.sum()

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.argmin(np.abs(cdf_ref - cdf_src[i]))

    return cv2.LUT(src, lut)


def histogram_matching_spec(src, ref):
    hist_src, _ = np.histogram(src.flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(ref.flatten(), 256, [0, 256])

    hist_src = hist_src / hist_src.sum()
    hist_ref = hist_ref / hist_ref.sum()

    cdf_src = np.cumsum(hist_src)
    cdf_ref = np.cumsum(hist_ref)

    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_ref[j] < cdf_src[i]:
            j += 1
        lut[i] = j

    return cv2.LUT(src, lut)


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


image = cv2.imread("images/deer.jpg", cv2.IMREAD_GRAYSCALE)

src_low = adjust_contrast(image, "low")
ref_high = adjust_contrast(image, "high")

res_cdf = histogram_matching_cdf(src_low, ref_high)
res_spec = histogram_matching_spec(src_low, ref_high)

display_images(
    [src_low, ref_high, res_cdf, res_spec],
    ["Source (Low)", "Reference (High)", "Matched (CDF)", "Matched (Spec)"],
)

print(f"Histogram Correlation - CDF:  {hist_corr(ref_high, res_cdf):.4f}")
print(f"Histogram Correlation - Spec: {hist_corr(ref_high, res_spec):.4f}")
