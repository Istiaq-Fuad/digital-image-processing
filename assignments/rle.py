import numpy as np

def rle_encode(image):
    pixels = image.flatten()
    encoded = []
    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1

    encoded.append((prev, count))
    return encoded

def rle_decode(encoded, shape):
    decoded = []
    for value, count in encoded:
        decoded.extend([value] * count)
    return np.array(decoded, dtype=np.uint8).reshape(shape)