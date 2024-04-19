import os
import shutil
import random


# input_folder = 'original_cleaned/archive/dataset'
input_folder = 'original_cleaned/archive/dataset'
output_folder = 'original_cleaned'

# Walk through all subfolders in the input folder
for dirpath, dirnames, filenames in os.walk(input_folder):
    # For each file in the current folder
    for filename in filenames:
        # Construct full file path
        file_path = os.path.join(dirpath, filename)
        renamed_file_path = os.path.join(dirpath, 'moved_'+dirpath.split('/')[-1]+f'_{random.randint(0, 1000)}_'+filename)
        # Move the file to the output folder
        shutil.move(file_path, renamed_file_path)
        shutil.move(renamed_file_path, output_folder)