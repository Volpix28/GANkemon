import numpy as np
from PIL import Image


def add_noise(image, noise_intensity_range):
    """
    Add a tiny amount of colorless noise to an image using Pillow.

    Parameters:
    image (PIL.Image.Image): The input image.
    noise_intensity_range (tuple): A tuple specifying the min and max values of noise intensity.

    Returns:
    PIL.Image.Image: The image with added noise.
    """
    # Convert the image to a numpy array
    img_array = np.array(image)

    # Get a random noise intensity between the specified range
    min_intensity, max_intensity = noise_intensity_range
    noise_intensity = np.random.uniform(min_intensity, max_intensity)

    # Generate noise with the chosen intensity
    noise = np.random.normal(0, noise_intensity, img_array.shape)

    # Add noise to the image
    noisy_img_array = img_array + noise

    # Clip the values to be within valid range (0 to 255) and convert to uint8
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # Convert the numpy array back to a Pillow image
    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img
