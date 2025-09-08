import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(img):

    hist, _ = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)

    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype("uint8")

    img_equalized = cdf_final[img]
    return img_equalized


img = cv2.imread("images/deer.jpg", cv2.IMREAD_GRAYSCALE)


custom_eq = histogram_equalization(img)


opencv_eq = cv2.equalizeHist(img)


cv2.imwrite("images/output/custom_equalized.jpg", custom_eq)
cv2.imwrite("images/output/opencv_equalized.jpg", opencv_eq)


fig, axes = plt.subplots(3, 2, figsize=(12, 12))


axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")


axes[0, 1].hist(img.flatten(), bins=256, range=[0, 256], color="gray")
axes[0, 1].set_title("Original Histogram")


axes[1, 0].imshow(custom_eq, cmap="gray")
axes[1, 0].set_title("Custom Equalized Image")
axes[1, 0].axis("off")


axes[1, 1].hist(custom_eq.flatten(), bins=256, range=[0, 256], color="gray")
axes[1, 1].set_title("Custom Histogram")


axes[2, 0].imshow(opencv_eq, cmap="gray")
axes[2, 0].set_title("OpenCV Equalized Image")
axes[2, 0].axis("off")


axes[2, 1].hist(opencv_eq.flatten(), bins=256, range=[0, 256], color="gray")
axes[2, 1].set_title("OpenCV Histogram")

plt.tight_layout()
plt.savefig("images/output/comparison_figure.png")
plt.show()
