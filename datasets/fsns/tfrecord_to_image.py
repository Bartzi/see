import argparse
import csv

import os
import re

import numpy as np
import tensorflow as tf

from PIL import Image


FILENAME_PATTERN = re.compile(r'.+-(\d+)-of-(\d+)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that takes tfrecord files and extracts all images + labels from it')
    parser.add_argument('tfrecord_dir', help='path to directory containing tfrecord files')
    parser.add_argument('destination_dir', help='path to dir where resulting images shall be saved')
    parser.add_argument('stage', help='stage of training these files are for [e.g. train]')

    args = parser.parse_args()

    os.makedirs(args.destination_dir, exist_ok=True)

    tfrecord_files = os.listdir(args.tfrecord_dir)
    tfrecord_files = sorted(tfrecord_files, key=lambda x: int(FILENAME_PATTERN.match(x).group(1)))

    with open(os.path.join(args.destination_dir, '{}.csv'.format(args.stage)), 'w') as label_file:
        writer = csv.writer(label_file, delimiter='\t')

        for tfrecord_file in tfrecord_files:
            tfrecord_filename = os.path.join(args.tfrecord_dir, tfrecord_file)

            file_id = FILENAME_PATTERN.match(tfrecord_file).group(1)
            dest_dir = os.path.join(args.destination_dir, args.stage, file_id)
            os.makedirs(dest_dir, exist_ok=True)

            record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_filename)

            for idx, string_record in enumerate(record_iterator):
                example = tf.train.Example()
                example.ParseFromString(string_record)

                labels = example.features.feature['image/class'].int64_list.value
                img_string = example.features.feature['image/encoded'].bytes_list.value[0]

                file_name = os.path.join(dest_dir, '{}.png'.format(idx))
                with open(file_name, 'wb') as f:
                    f.write(img_string)

                label_file_data = [file_name]
                label_file_data.extend(labels)
                writer.writerow(label_file_data)
                print("recovered {:0>6} files".format(idx), end='\r')
