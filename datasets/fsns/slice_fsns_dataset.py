import argparse
import csv

import os
from PIL import Image


def find_way_to_common_dir(common_dir, dir):
    dirs = []
    base_dir = dir
    while True:
        base_dir, dirname = os.path.split(base_dir)
        dirs.append(dirname)
        if base_dir in common_dir:
            break

    return reversed(dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that slice one FSNS image to 4 slices')
    parser.add_argument('label_file', help='path to label file that has path to image + labels')
    parser.add_argument('destination_dir', help='path to dir where resulting images shall be put')
    parser.add_argument('-b', '--base-dir', help='path to base dir of every file in label file')

    args = parser.parse_args()

    label_file_name = os.path.basename(args.label_file)
    label_dir = os.path.dirname(args.label_file)

    with open(args.label_file) as label_file, open(os.path.join(args.destination_dir, label_file_name), 'w') as dest_file:
        reader = csv.reader(label_file, delimiter='\t')
        writer = csv.writer(dest_file, delimiter='\t')

        for idx, info in enumerate(reader):
            image_path = info[0]
            labels = info[1:]
            image = Image.open(image_path)
            image = image.convert('RGB')

            save_dir = os.path.join(args.destination_dir, *find_way_to_common_dir(label_dir, os.path.dirname(image_path)))
            os.makedirs(save_dir, exist_ok=True)

            image_name = os.path.splitext(os.path.basename(image_path))[0]

            for i in range(4):
                part_image = image.crop((i * 150, 0, (i + 1) * 150, 150))
                file_name = "{}_{}.png".format(os.path.join(save_dir, image_name), i)
                part_image.save(file_name)
                writer.writerow((file_name, *labels))

            print("done with {:6} files".format(idx), end='\r')
