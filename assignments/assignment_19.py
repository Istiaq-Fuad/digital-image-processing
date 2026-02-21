import numpy as np


def quantize_332(image):
    r = (image[:, :, 0] >> 5) << 5
    g = (image[:, :, 1] >> 5) << 5
    b = (image[:, :, 2] >> 6) << 6

    quantized = np.stack([r, g, b], axis=2)
    return quantized.astype(np.uint8)


def reduce_bit_depth(image, bits):
    levels = 2**bits
    factor = 256 // levels
    reduced = (image // factor) * factor
    return reduced.astype(np.uint8)
