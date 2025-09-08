import cv2
import numpy as np
import matplotlib.pyplot as plt


def bit_plane_slicing(img):
    planes = []
    for i in range(8):

        plane = (img & (1 << i)) >> i
        plane_img = plane * (2**i)
        planes.append(plane_img)
    return planes


def reconstruct_from_planes(planes):
    reconstructed = np.zeros_like(planes[0], dtype=np.uint8)
    for p in planes:
        reconstructed += p
    return reconstructed


def display_planes(original, planes):
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 5, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i, plane in enumerate(planes):
        plt.subplot(3, 5, i + 2)
        plt.imshow(plane, cmap="gray")
        plt.title(f"Bit Plane {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = "test.jpg"
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    planes = bit_plane_slicing(gray_img)

    reconstructed_img = reconstruct_from_planes(planes)
    loss = np.sum(gray_img - reconstructed_img)

    print(f"Reconstruction Loss: {loss}")
    if loss == 0:
        print("Lossless reconstruction successful!")

    display_planes(gray_img, planes)
