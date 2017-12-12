import argparse
import csv
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="take svhn gt json and create csv file")
    parser.add_argument('json_gt', help='path to svhn json gt')
    parser.add_argument('destination', help='path to resulting gt file in csv format')
    parser.add_argument('--max-length', type=int, help='maximum length of labels [default: take max length found]')

    args = parser.parse_args()

    with open(args.json_gt) as gt_file:
        gt = json.load(gt_file)

    # read information for all files
    base_dir = os.path.abspath(os.path.dirname(args.json_gt))
    file_info = []
    for image_data in gt:
        filename = os.path.join(base_dir, image_data['filename'])
        labels = [int(b['label']) if int(b['label']) != 10 else 0 for b in image_data['boxes']]
        file_info.append((filename, labels))

    # determine max length of labels
    if args.max_length is None:
        max_length = max(map(lambda x: len(x[1]), file_info))
    else:
        max_length = args.max_length

    # pad and filter labels
    filtered_infos = []
    for file_path, labels in file_info:
        if len(labels) > max_length:
            continue
        elif len(labels) < max_length:
            values_to_pad = [10] * (max_length - len(labels))
            labels.extend(values_to_pad)
        file_info = [file_path] + labels
        filtered_infos.append(file_info)

    # write to csv file
    with open(args.destination, 'w') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerows(filtered_infos)
