import numpy as np


def add_gaussian_noise(image, mean=0, sigma=25):
    image = image.astype(np.float32)

    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise

    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)


def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.copy()

    # Salt
    salt = np.random.rand(*image.shape) < prob
    noisy[salt] = 255

    # Pepper
    pepper = np.random.rand(*image.shape) < prob
    noisy[pepper] = 0

    return noisy
