from scipy.stats import wasserstein_distance
import numpy as np
import os
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            images.append(np.array(img).flatten())
    return images

# Load images from two datasets
dataset1 = load_images_from_folder('dataset1')
dataset2 = load_images_from_folder('dataset2')

# Flatten the datasets into 1-dimensional distributions
dataset1_flat = np.array(dataset1).flatten()
dataset2_flat = np.array(dataset2).flatten()

# Compute the Wasserstein distance between the two datasets
w_dist = wasserstein_distance(dataset1_flat, dataset2_flat)

print(f"The Wasserstein distance between the two datasets is: {w_dist}")