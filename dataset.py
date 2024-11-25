import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to root 'LCC_FASD' folder")
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to output resized dataset folder")
parser.add_argument("-s", "--target_size", type=int, default=224, help="Target size for resized images")

def resize_image_with_aspect_ratio(image, target_size=224, padding_color=(0, 0, 0)):
    current_height, current_width = image.shape[:2]
    aspect_ratio = current_width / float(current_height)

    if aspect_ratio > 1:  # Horizontal image
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
        pad_height = target_size - new_height
        top_pad = pad_height // 2
        padded_image = np.full((target_size, new_width, 3), padding_color, dtype=np.uint8)
        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image[top_pad:top_pad + new_height, :, :] = resized_image
    else:  # Vertical image
        new_height = target_size
        new_width = int(new_height * aspect_ratio)
        pad_width = target_size - new_width
        left_pad = pad_width // 2
        padded_image = np.full((new_height, target_size, 3), padding_color, dtype=np.uint8)
        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image[:, left_pad:left_pad + new_width, :] = resized_image

    return padded_image

def process_folder(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    for subset in ['LCC_FASD_training', 'LCC_FASD_development', 'LCC_FASD_evaluation']:
        subset_path = os.path.join(input_dir, subset)
        output_subset_path = os.path.join(output_dir, subset)
        os.makedirs(output_subset_path, exist_ok=True)

        for label in ['real', 'spoof']:
            input_label_path = os.path.join(subset_path, label)
            output_label_path = os.path.join(output_subset_path, label)
            os.makedirs(output_label_path, exist_ok=True)

            print(f"Processing {label} data in {subset}...")
            for file_name in tqdm(os.listdir(input_label_path)):
                file_path = os.path.join(input_label_path, file_name)

                # Read image
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error reading file {file_path}, skipping.")
                    continue

                # Resize image
                resized_image = resize_image_with_aspect_ratio(image, target_size)

                # Save resized image
                output_file_path = os.path.join(output_label_path, file_name)
                cv2.imwrite(output_file_path, resized_image)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    target_size = args["target_size"]

    process_folder(input_dir, output_dir, target_size)
