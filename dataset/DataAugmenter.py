import os
import random
from functools import reduce

from PIL import Image, ImageEnhance
from tqdm import tqdm

from dataset.DataTransformer import DataTransformer
from utils.add_noise import add_noise


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
            # Open the image
            img_path = os.path.join(self.src_folder, img_file)
            img = Image.open(img_path)

            # Randomly choose augmentation techniques
            augmentation_functions = [
                lambda img: img.rotate(random.randint(-15, 15), fillcolor="white"),
                lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.75, 1.25)),
                lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(1, 1.25)),
                lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25)),
                lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5)),
                lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            ]

            num_augmentations = random.randint(1, len(augmentation_functions))
            for _ in range(num_augmentations):
                selected_functions = random.sample(augmentation_functions, 3)
                img = reduce(lambda x, func: func(x), selected_functions, img)

            img = add_noise(img, (0.1, 0.2))
            output_path = os.path.join(self.output_folder, f"{os.path.splitext(img_file)[0]}_augmented.jpg")
            img.save(output_path)


augmenter = DataAugmenter("nested/transformed_data", "nested/augmented_data", 42)
augmenter.augment_data()
