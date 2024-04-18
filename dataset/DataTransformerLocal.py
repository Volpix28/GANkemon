import os
from PIL import Image, ImageOps

def transform_images_from_folder(input_folder, output_folder):
    """
    Transform all images in the input folder by making them square and resizing them, 
    then save them to the output folder.
    Args:
    input_folder (str): Folder path where the input images are stored.
    output_folder (str): Folder path where the transformed images should be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGBA')

        max_dim = max(img.width, img.height)
        square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))

        offset = ((max_dim - img.width) // 2, (max_dim - img.height) // 2)
        square_img.paste(img, offset, img)  # Use img as mask to handle transparency

        resized_img = square_img.resize((256, 256), Image.Resampling.LANCZOS)

        output_filename = os.path.join(output_folder, f"transformed_{i}.png")
        resized_img.save(output_filename)
        print(f"Image {i}: saved to {output_filename}")

# Example usage
input_folder = "dataset/dataset_2"
output_folder = "dataset/transformed_data_2"
transform_images_from_folder(input_folder, output_folder)
