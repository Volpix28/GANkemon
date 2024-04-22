from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def ssim_metric(path1, path2):
    # Load images from two datasets
    dataset1 = load_images_from_folder(path1)
    dataset2 = load_images_from_folder(path2)

    # For each image in dataset 2, find the most similar image in dataset 1
    ssims =[]
    for i, img2 in enumerate(dataset2):
        max_ssim = -1
        max_index = -1
        for j, img1 in enumerate(dataset1):
            s = return ssim(img1, img2, data_range=255)
            if s > max_ssim:
                max_ssim = s
                max_index = j
        print(f"Image {i} in dataset 2 is most similar to image {max_index} in dataset 1 with SSIM: {max_ssim}")
        ssims.append(max_ssim)

    # Compute the average SSIM between the two datasets
    avg_ssim = np.mean(ssims)
    print(f"The average SSIM between the two datasets is: {avg_ssim}")
    return avg_ssim

path1 = './dataset/bulbapedia/transformed'
path2 = './outputs/23_first_try_bulbapedia_23_bulbapedia_256x256_e50/step5'

ssim_value = ssim_metric(path1, path2)

print(ssim_value)