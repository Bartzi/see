import argparse
import os
import shutil
from PIL import Image

from create_svhn_dataset_4_images import SUPPORTED_IMAGE_TYPES


def get_images(image_dir, images, min_width, min_height):
    for an_image in images:
        with Image.open(os.path.join(image_dir, an_image)) as image:
            width, height = image.size

        if width >= min_width and height >= min_height:
            yield an_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that filters all images larger than a given size from a directory')
    parser.add_argument('image_dir', help='path to dir containing images to filter')
    parser.add_argument('destination_dir', help='path to dir where filtered images shall be put')
    parser.add_argument('min_width', type=int, help='minimum width of images that shall be filtered')
    parser.add_argument('min_height', type=int, help='minimum height of images that shall be filtered')

    args = parser.parse_args()

    images = filter(lambda x: os.path.splitext(x)[-1] in SUPPORTED_IMAGE_TYPES, os.listdir(args.image_dir))

    for image in get_images(args.image_dir, images, args.min_width, args.min_height):
        shutil.copy(os.path.join(args.image_dir, image), os.path.join(args.destination_dir, image))
