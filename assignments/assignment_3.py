import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_bit_planes(image):
    bit_planes = []

    for i in range(8):
        plane = (image >> i) & 1
        plane = plane * 255
        bit_planes.append(plane)

    return bit_planes


def merge_all_planes(bit_planes):
    merged = np.zeros_like(bit_planes[0])
    for i in range(8):
        merged += (bit_planes[i] // 255) << i

    return merged


def merge_partial_plane(bit_planes, indices):
    merged = np.zeros_like(bit_planes[0])
    for i in indices:
        merged += (bit_planes[i] // 255) << i

    return merged


def main():
    image = cv2.imread("images/image1.jpg", 0)

    bit_planes = get_bit_planes(image)

    merged_image = merge_all_planes(bit_planes)

    partial_1 = merge_partial_plane(bit_planes, [0, 2, 4])
    partial_2 = merge_partial_plane(bit_planes, [1, 5, 6, 7])

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i in range(8):
        plt.subplot(4, 4, i + 2)
        plt.imshow(bit_planes[i], cmap="gray")
        plt.title("Bit Plane " + str(i))
        plt.axis("off")

    plt.subplot(4, 4, 11)
    plt.imshow(merged_image, cmap="gray")
    plt.title("Merged")
    plt.axis("off")

    plt.subplot(4, 4, 12)
    plt.imshow(partial_1, cmap="gray")
    plt.title("Merged: 0 2 4")
    plt.axis("off")

    plt.subplot(4, 4, 13)
    plt.imshow(partial_2, cmap="gray")
    plt.title("Merged: 1,5,6,7")
    plt.axis("off")

    plt.tight_layout()
    import os

    os.makedirs("output_images", exist_ok=True)
    plt.savefig("output_images/bit_planes.png")


if __name__ == "__main__":
    main()
