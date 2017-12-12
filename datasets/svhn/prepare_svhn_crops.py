import argparse
import csv
import json
import os
import tqdm as tqdm
from PIL import Image

from create_svhn_dataset import BBox


def merge_bboxes(bboxes):
    resulting_bbox = None
    for bbox in bboxes:
        if resulting_bbox is None:
            resulting_bbox = bbox
            continue

        max_right = max(resulting_bbox.left + resulting_bbox.width, bbox.left + bbox.width)
        max_bottom = max(resulting_bbox.top + resulting_bbox.height, bbox.top + bbox.height)

        resulting_bbox.top = min(resulting_bbox.top, bbox.top)
        resulting_bbox.left = min(resulting_bbox.left, bbox.left)
        resulting_bbox.width = max_right - resulting_bbox.left
        resulting_bbox.height = max_bottom - resulting_bbox.top
        resulting_bbox.label.extend(bbox.label)

    return resulting_bbox


def enlarge_bbox(bbox, percentage=0.3):
    enlarge_width = bbox.width * percentage * 2
    enlarge_height = bbox.height * percentage * 2

    return BBox(
        label=bbox.label,
        left=bbox.left - enlarge_width // 2,
        width=bbox.width + enlarge_width,
        top=bbox.top - enlarge_height // 2,
        height=bbox.height + enlarge_height,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that takes user defined crops around image regions of svhn data')
    parser.add_argument('svhn_json', help='path to svhn gt file')
    parser.add_argument('crop_size', type=int, help='crop size in pixels for each text region')
    parser.add_argument('dest_dir', help='destination dir for cropped images')
    parser.add_argument('stage_name', help='name of stage (e.g. train, or test)')
    parser.add_argument('--max-length', type=int, default=5, help='max length of labels [default: 5]')

    args = parser.parse_args()

    with open(args.svhn_json) as gt_file:
        gt = json.load(gt_file)

    # create dest dir if it does not exist
    os.makedirs(args.dest_dir, exist_ok=True)

    # read information for all files
    base_dir = os.path.abspath(os.path.dirname(args.svhn_json))
    file_info = []
    for image_data in tqdm.tqdm(gt):
        filename = os.path.join(base_dir, image_data['filename'])
        bboxes = [BBox(
            label=int(b['label']),
            top=b['top'],
            height=b['height'],
            left=b['left'],
            width=b['width'],
        ) for b in image_data['boxes']]

        merged_bbox = merge_bboxes(bboxes)
        new_bbox = enlarge_bbox(merged_bbox)

        with Image.open(filename) as image:
            cropped_image = image.crop((
                new_bbox.left,
                new_bbox.top,
                new_bbox.left + new_bbox.width,
                new_bbox.top + new_bbox.height,
            ))
            cropped_image = cropped_image.resize((args.crop_size, args.crop_size))
            new_filename = os.path.join(args.dest_dir, image_data['filename'])
            cropped_image.save(new_filename)
        file_info.append((os.path.abspath(new_filename), new_bbox.label))

    # pad and filter labels
    filtered_infos = []
    for file_path, labels in file_info:
        if len(labels) > args.max_length:
            continue
        elif len(labels) < args.max_length:
            values_to_pad = [0] * (args.max_length - len(labels))
            labels.extend(values_to_pad)
        file_info = [file_path] + labels
        filtered_infos.append(file_info)

    # write to csv file
    with open(os.path.join(os.path.dirname(args.dest_dir), "{}.csv".format(args.stage_name)), 'w') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerows(filtered_infos)