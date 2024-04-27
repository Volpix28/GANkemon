from datasets import load_dataset
import os
import glob
from tqdm import tqdm
from PIL import Image


class DataTransformer:

    def transform_remote(self, output_folder, dataset_name="huggan/pokemon"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dataset = load_dataset(dataset_name, split="train")

        offset = len(glob.glob(output_folder + "*.png"))

        for i, example in enumerate(tqdm(dataset, desc="Processing images", unit="images")):
            img = example["image"]

            resized_img = self._resize_image(img)

            filename = os.path.join(output_folder, f"pokemon_{i+offset}.png")
            resized_img.save(filename)

    def transform_local(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if
                       f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"))]

        offset = len(glob.glob(output_folder + "*.png"))

        for i, filename in enumerate(tqdm(image_files, desc="Processing images", unit="images")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGBA")

            resized_img = self._resize_image(img)

            output_filename = os.path.join(output_folder, f"pokemon_{i+offset}.png")
            resized_img.save(output_filename)

    def _resize_image(self, image):
        max_dim = max(image.width, image.height)

        # Create a new image with the same size as the original image and white background
        white_img = Image.new("RGBA", (max_dim, max_dim), "WHITE")

        # Paste the original image onto the white image, using the original image's alpha channel as the mask
        white_img.paste(image, (0, 0), image)

        # Convert the image back to RGB
        rgb_img = white_img.convert("RGB")

        resized_img = rgb_img.resize((256, 256), Image.Resampling.LANCZOS)

        return resized_img
    

if __name__ == "__main__":
    transformer = DataTransformer()
    transformer.transform_local("../dataset/all_assets_raw", "../dataset/all_assets/transformed")