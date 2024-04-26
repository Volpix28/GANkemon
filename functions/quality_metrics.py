import os
import random
import numpy as np
from PIL import Image
import tempfile
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from pytorch_fid import fid_score
from typing_extensions import deprecated


def load_images_from_folder(folder, flattening_required=False):
    """_summary_

    Args:
        folder (str): Path to folder with images
        flattening_required (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
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


def resize_to_smaller(img1, img2):
    # Determine the size of both images
    size1 = img1.size
    size2 = img2.size

    # If img1 is larger than img2, resize img1
    if size1[0] > size2[0] or size1[1] > size2[1]:
        img1 = img1.resize(size2, Image.LANCZOS)
    # If img2 is larger than img1, resize img2
    elif size2[0] > size1[0] or size2[1] > size1[1]:
        img2 = img2.resize(size1, Image.LANCZOS)

    # Convert the images to numpy arrays
    img1 = np.array(img1)
    img2 = np.array(img2)

    return img1, img2


@deprecated
def nn_metric(path1, path2, verbose=False):
    """_summary_

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset

    Returns:
        _type_: _description_
    """
    # Load images from two datasets
    if type(path1) == str:
        dataset1 = load_images_from_folder(path1, flattening_required=True)
    else:
        dataset1 = path1
    if type(path2) == str:
        dataset2 = load_images_from_folder(path2, flattening_required=True)
    else:
        dataset2 = path2

    # Fit the NearestNeighbors model to dataset 1
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(dataset1)

    # Find the nearest neighbor in dataset 1 for each image in dataset 2
    distances, indices = nbrs.kneighbors(dataset2)

    # Print the smallest distance and its corresponding index in dataset 1 for each image in dataset 2
    distances = []
    for i in range(len(dataset2)):
        if verbose == True:
            print(
                f"Image {i} in dataset 2 is closest to image {indices[i][0]} in dataset 1 with distance: {distances[i][0]}")
        distances.append(distances[i][0])

    # Compute the average distance between the two datasets
    avg_distance = np.mean(distances)
    if verbose == True:
        print(
            f"The average distance between the two datasets is: {avg_distance}")
    return avg_distance


def ssim_metric(path1, path2, verbose=False, sample=None):
    """Average Structural Similarity Index (SSIM) between two datasets.
    SSIM is between 0 and 1, where 1 means the images are identical.
    For more information see: https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset

    Returns:
        _type_: _description_
    """
    # Load images from two datasets
    if type(path1) == str:
        dataset1 = load_images_from_folder(path1)
    else:
        dataset1 = path1
    if type(path2) == str:
        dataset2 = load_images_from_folder(path2)
    else:
        dataset2 = path2

    if sample != None and len(dataset1) > sample:
        dataset1 = random.sample(dataset1, sample)

    # For each image in dataset 2, find the most similar image in dataset 1
    ssims = []
    for i, img2 in enumerate(dataset2):
        max_ssim = -1
        max_index = -1
        for j, img1 in enumerate(dataset1):
            rimg1, rimg2 = resize_to_smaller(
                Image.fromarray(img1), Image.fromarray(img2))
            s = ssim(rimg1, rimg2, data_range=255, channel_axis=2)
            if s > max_ssim:
                max_ssim = s
                max_index = j
        if verbose == True:
            print(
                f"Image {i} in dataset 2 is most similar to image {max_index} in dataset 1 with SSIM: {max_ssim}")
        ssims.append(max_ssim)

    # Compute the average SSIM between the two datasets
    avg_ssim = np.mean(ssims)
    if verbose == True:
        print(f"The average SSIM between the two datasets is: {avg_ssim}")
    return avg_ssim


def waterstein_distance_metric(path1, path2, batch_size=200, verbose=False, sample=None):
    """_summary_

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset

    Returns:
        _type_: _description_
    """
    # Load images from two datasets
    if type(path1) == str:
        dataset1 = load_images_from_folder(path1, flattening_required=True)
    else:
        dataset1 = path1
    if type(path2) == str:
        dataset2 = load_images_from_folder(path2, flattening_required=True)
    else:
        dataset2 = path2

    # # Flatten the datasets into 1-dimensional distributions
    # dataset1_flat = np.array(dataset1).flatten() #again???
    dataset2_flat = np.array(dataset2).flatten()

    if sample != None and len(dataset1) > sample:
        dataset1 = random.sample(dataset1, sample)

    w_dist = 0
    start = 0
    n_batches = int(np.ceil(len(dataset1) / float(batch_size)))
    end = batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch1 = np.array(dataset1[start:end]).flatten()
        w_dist += wasserstein_distance(batch1, dataset2_flat)
    w_dist /= n_batches

    # Compute the Wasserstein distance between the two datasets
    # w_dist = wasserstein_distance(dataset1_flat, dataset2_flat) #wasserstein_distance_nd maybe?

    if verbose == True:
        print(
            f"The Wasserstein distance between the two datasets is: {w_dist}")
    return w_dist


def fid_metric(path1, path2, batch_size=32, dims=64, device="cpu"):
    """Calculate the Frechet Inception Distance (FID) between two datasets.

    Args:
        path1 (str): Path to the original Dataset
        path2 (str): Path to the generated Dataset
        batch_size (int, optional): _description_. Defaults to 32.
        dims (int, optional): 64: first max pooling features, 192: second max pooling features, 768: pre-aux classifier features, 2048: final average pooling features (this is the default). Defaults to 64.
        device (str, optional): Device to which pytroch should delegate the calculations. Defaults to "cuda".

    Returns:
        _type_: FID value
    """

    cleanup_path1 = False
    cleanup_path2 = False

    if type(path1) != str:
        dataset = path1
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        for i, img in enumerate(dataset):
            # Create a temporary file in the temporary directory
            temp_file_path = os.path.join(temp_dir, f"temp_image_{i}.png")
            # Save the image to the temporary file
            im = Image.fromarray(img)
            im.save(temp_file_path)
        # Replace dataset with the list of temporary file paths
        path1 = temp_dir
        cleanup_path1 = True

    if type(path2) != str:
        dataset = path2
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        for i, img in enumerate(dataset):
            # Create a temporary file in the temporary directory
            temp_file_path = os.path.join(temp_dir, f"temp_image_{i}.png")
            # Save the image to the temporary file
            im = Image.fromarray(img)
            im.save(temp_file_path)
        # Replace dataset with the list of temporary file paths
        path2 = temp_dir
        cleanup_path2 = True

    print(fid_score)
    fid_value = fid_score.calculate_fid_given_paths(
        [path1, path2], batch_size, device, dims,
    )

    # Clean up the temporary files
    if cleanup_path1:
        for path in dataset1:
            os.remove(path)
    if cleanup_path2:
        for path in dataset1:
            os.remove(path)

    return fid_value


if __name__ == "__main__":
    dataset1 = "./dataset/bulbapedia/transformed"
    dataset2 = "./outputs/23_first_try_bulbapedia_23_bulbapedia_256x256_e50/step5"

    print(nn_metric(dataset1, dataset2))
    print(ssim_metric(dataset1, dataset2))
    print(waterstein_distance_metric(dataset1, dataset2))
    print(fid_metric(dataset1, dataset2))
