import argparse
import os

import tqdm

from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="path to directory with source files")
    parser.add_argument("dest_dir", help="path to directory with cropped files")
    parser.add_argument("crop_window", type=int, nargs="*", default=[150, 0, 750, 300], help="crop window in (x, y, x, y) starting at top left and then bottom right")

    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)
    for image_name in tqdm.tqdm(os.listdir(args.source_dir)):
        image_path = os.path.join(args.source_dir, image_name)
        with Image.open(image_path) as the_image:
            the_image = the_image.crop(args.crop_window)
            the_image.save(os.path.join(args.dest_dir, image_name))
