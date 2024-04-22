from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from pytorch_fid import fid_score
import numpy as np
import os
from PIL import Image

def load_images_from_folder(folder, flattening_required = False):
    """_summary_

    Args:
        folder (str): Path to folder with images
        flattening_required (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            if flattening_required:
                images.append(np.array(img).flatten())
            else:
                images.append(np.array(img))
    return images

def nn_metirc(path1, path2):
    """_summary_

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset

    Returns:
        _type_: _description_
    """
    # Load images from two datasets
    dataset1 = load_images_from_folder(path1, flattening_required=True)
    dataset2 = load_images_from_folder(path2, flattening_required=True)

    # Fit the NearestNeighbors model to dataset 1
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(dataset1)

    # Find the nearest neighbor in dataset 1 for each image in dataset 2
    distances, indices = nbrs.kneighbors(dataset2)

    # Print the smallest distance and its corresponding index in dataset 1 for each image in dataset 2
    distances = []
    for i in range(len(dataset2)):
        print(f"Image {i} in dataset 2 is closest to image {indices[i][0]} in dataset 1 with distance: {distances[i][0]}")
        distances.append(distances[i][0])
    
    # Compute the average distance between the two datasets
    avg_distance = np.mean(distances)
    print(f"The average distance between the two datasets is: {avg_distance}")
    return avg_distance

def ssim_metric(path1, path2):
    """_summary_

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset

    Returns:
        _type_: _description_
    """
    # Load images from two datasets
    dataset1 = load_images_from_folder(path1) #TODO: eventually needs grayscale?
    dataset2 = load_images_from_folder(path2)

    # For each image in dataset 2, find the most similar image in dataset 1
    ssims =[]
    for i, img2 in enumerate(dataset2):
        max_ssim = -1
        max_index = -1
        for j, img1 in enumerate(dataset1):
            s = ssim(img1, img2, data_range=255, channel_axis=2)
            if s > max_ssim:
                max_ssim = s
                max_index = j
        print(f"Image {i} in dataset 2 is most similar to image {max_index} in dataset 1 with SSIM: {max_ssim}")
        ssims.append(max_ssim)

    # Compute the average SSIM between the two datasets
    avg_ssim = np.mean(ssims)
    print(f"The average SSIM between the two datasets is: {avg_ssim}")
    return avg_ssim

def waterstein_distance_metric(path1, path2):
    """_summary_

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset

    Returns:
        _type_: _description_
    """
    # Load images from two datasets
    dataset1 = load_images_from_folder(path1, flattening_required=True)
    dataset2 = load_images_from_folder(path2, flattening_required=True)

    # Flatten the datasets into 1-dimensional distributions
    dataset1_flat = np.array(dataset1).flatten()
    dataset2_flat = np.array(dataset2).flatten()

    # Compute the Wasserstein distance between the two datasets
    w_dist = wasserstein_distance(dataset1_flat, dataset2_flat)

    print(f"The Wasserstein distance between the two datasets is: {w_dist}")
    return w_dist


def fid_metirc(path1, path2, batch_size = 32, dims = 64, device = 'cuda'):
    """Calculate the Frechet Inception Distance (FID) between two datasets.

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset
        batch_size (int, optional): _description_. Defaults to 32.
        dims (int, optional): 64: first max pooling features, 192: second max pooling features, 768: pre-aux classifier features, 2048: final average pooling features (this is the default). Defaults to 64.
        device (str, optional): Device to which pytroch should delegate the calculations. Defaults to 'cuda'.

    Returns:
        _type_: FID value
    """
    # 64: first max pooling features
    # 192: second max pooling features
    # 768: pre-aux classifier features
    # 2048: final average pooling features (this is the default)

    print(fid_score)
    fid_value = fid_score.calculate_fid_given_paths(
            [path1, path2], batch_size, device, dims,
    )
    return fid_value

if __name__ == '__main__':
    dataset1 = './dataset/bulbapedia/transformed'
    dataset2 = './outputs/23_first_try_bulbapedia_23_bulbapedia_256x256_e50/step5'

    print(nn_metirc(dataset1, dataset2))
    print(ssim_metric(dataset1, dataset2))
    print(waterstein_distance_metric(dataset1, dataset2))
    print(fid_metirc(dataset1, dataset2))