import os
from PIL import Image


def load_images_from_folder(folder, num_images=100):
    images = []
    for i in range(0, num_images):
        filename = os.path.join(folder, f"img_{i}.png")
        try:
            img = Image.open(filename)
            images.append(img)
        except IOError:
            print(f"Error opening {filename}")
    return images
