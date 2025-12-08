import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


def dft_1d_naive(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def dft_2d_naive(image):
    h, w = image.shape

    row_transform = np.zeros((h, w), dtype=complex)
    for i in range(h):
        row_transform[i, :] = dft_1d_naive(image[i, :])

    col_transform = np.zeros((h, w), dtype=complex)
    for j in range(w):
        col_transform[:, j] = dft_1d_naive(row_transform[:, j])

    return col_transform


def fft_1d_recursive(x):
    N = len(x)

    if N <= 1:
        return x

    even = fft_1d_recursive(x[0::2])
    odd = fft_1d_recursive(x[1::2])

    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

    return [even[k] + T[k] for k in range(N // 2)] + [
        even[k] - T[k] for k in range(N // 2)
    ]


def fft_2d_custom(image):
    h, w = image.shape

    row_fft = np.zeros((h, w), dtype=complex)
    for i in range(h):
        row_fft[i, :] = fft_1d_recursive(image[i, :])

    col_fft = np.zeros((h, w), dtype=complex)
    temp = row_fft.T
    for i in range(w):
        col_fft[:, i] = fft_1d_recursive(temp[i, :])

    return col_fft.T


sizes = [16, 32, 64]

dft_times = []
fft_times = []

print("Starting Benchmark...")

for N in sizes:
    print(f"Processing image size: {N}x{N}...")

    img = np.random.rand(N, N)

    start = time.time()
    dft_2d_naive(img)
    dft_times.append(time.time() - start)

    start = time.time()
    fft_2d_custom(img)
    fft_times.append(time.time() - start)

large_sizes = [128, 256, 512]
fft_large_times = []
for N in large_sizes:
    print(f"Processing large image size (FFT only): {N}x{N}...")
    img = np.random.rand(N, N)
    start = time.time()
    fft_2d_custom(img)
    fft_large_times.append(time.time() - start)

plt.figure(figsize=(10, 6))

plt.plot(sizes, dft_times, "r-o", label="Naive DFT Implementation", linewidth=2)
plt.plot(sizes, fft_times, "b-o", label="Recursive FFT Implementation", linewidth=2)

plt.plot(large_sizes, fft_large_times, "b--", label="FFT (Large Images)")

plt.title("Performance Comparison: Naive DFT vs. Recursive FFT")
plt.xlabel("Image Width (N pixels)")
plt.ylabel("Execution Time (seconds)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

plt.annotate(
    "DFT Cost Explosion",
    xy=(64, dft_times[-1]),
    xytext=(80, dft_times[-1]),
    arrowprops=dict(facecolor="black", shrink=0.05),
)

plt.tight_layout()
plt.savefig("images/output/dft_vs_fft_benchmark.png")
print("Benchmark complete. Figure saved as 'dft_vs_fft_benchmark.png'")
