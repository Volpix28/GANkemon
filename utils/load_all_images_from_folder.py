import os

from PIL import Image
import numpy as np


def load_all_images_from_folder(folder, flattening_required=False):
    """
    Load all images in the provided folder.
    """
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            if flattening_required:
                images.append(np.array(img).flatten())
            else:
                images.append(np.array(img))
    return images
