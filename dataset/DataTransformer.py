from PIL.Image import Resampling
from datasets import load_dataset
import os
from tqdm import tqdm
from PIL import Image


class DataTransformer:

    def transform_remote(self, output_folder, dataset_name="huggan/pokemon"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dataset = load_dataset(dataset_name, split="train")

        for i, example in enumerate(tqdm(dataset, desc="Processing images", unit="images")):
            img = example["image"]

            resized_img = self._resize_image(img)

            filename = os.path.join(output_folder, f"pokemon_{i}.png")
            resized_img.save(filename)

    def transform_local(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

        for i, filename in enumerate(tqdm(image_files, desc="Processing images", unit="images")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('RGBA')

            resized_img = self._resize_image(img)

            output_filename = os.path.join(output_folder, f"transformed_{i}.png")
            resized_img.save(output_filename)

    def _resize_image(self, image):
        max_dim = max(image.width, image.height)

        square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))

        offset = ((max_dim - image.width) // 2, (max_dim - image.height) // 2)
        square_img.paste(image, offset)

        resized_img = square_img.resize((256, 256), Resampling.LANCZOS)

        return resized_img
