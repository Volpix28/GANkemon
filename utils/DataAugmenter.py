import os
import random
from functools import reduce

from PIL import Image, ImageEnhance
from tqdm import tqdm

from add_noise import add_noise


class DataAugmenter:

    def __init__(self, src_folder, output_folder, seed):
        self.src_folder = src_folder
        self.output_folder = output_folder
        self.seed = seed

    def augment_data(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        image_files = [f for f in os.listdir(self.src_folder) if f.endswith(".png")]

        for _, img_file in enumerate(tqdm(image_files, desc="Processing images", unit="images")):
            img_path = os.path.join(self.src_folder, img_file)

            img = Image.open(img_path)
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = add_noise(img, (1, 3))

            img_two = Image.open(img_path)
            img_two = img_two.rotate(random.randint(-15, 15), fillcolor="white")
            img_two = ImageEnhance.Sharpness(img_two).enhance(random.uniform(0.8, 1.2))
            img_two = add_noise(img_two, (1, 3))

            output_path = os.path.join(self.output_folder, f"{os.path.splitext(img_file)[0]}_augmented_1.png")
            output_path_two = os.path.join(self.output_folder, f"{os.path.splitext(img_file)[0]}_augmented_2.png")

            img.save(output_path)
            img_two.save(output_path_two)


augmenter = DataAugmenter("../dataset/all_assets/transformed", "../dataset/all_assets/augmented", 42)
augmenter.augment_data()
