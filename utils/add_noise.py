import numpy as np
from PIL import Image


def add_noise(image, noise_intensity_range):
    """
    Add a tiny amount of colorless noise to an image using Pillow.
    """
    img_array = np.array(image)

    min_intensity, max_intensity = noise_intensity_range
    noise_intensity = np.random.uniform(min_intensity, max_intensity)

    noise = np.random.normal(0, noise_intensity, img_array.shape)

    noisy_img_array = img_array + noise

    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img
