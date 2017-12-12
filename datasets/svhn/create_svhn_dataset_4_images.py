import argparse
import csv
import json

import itertools
import os
import random

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_TYPES = ['.png']


class SVHNDatasetCreator:

    def __init__(self, image_dir, image_size, image_columns, image_rows, destination_dir, dataset_name, max_label_length, label_pad_value=10):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_columns = image_columns
        self.images_per_sample = image_columns * image_rows
        self.destination_dir = destination_dir
        self.dataset_name = dataset_name
        self.max_label_length = max_label_length

        self.destination_image_dir = os.path.join(destination_dir, dataset_name)
        os.makedirs(self.destination_image_dir, exist_ok=True)
        self.label_pad_value = label_pad_value

        self.interpolation_area = 0.15
        self.image_rows = image_rows
        self.dest_image_size = (
            int(image_rows * self.image_size - (image_rows - 1) * self.image_size * self.interpolation_area),
            int(image_columns * self.image_size - (image_columns - 1) * self.image_size * self.interpolation_area),
        )
        self.overlap_map = np.zeros(self.dest_image_size, dtype=np.uint32)

    def blend_at_intersection(self, images):
        pre_blend_images = []
        for idx, image in enumerate(images):
            x_start, y_start = self.get_start_indices(idx)
            image_data = np.asarray(image)
            overlap_data = self.overlap_map[y_start:y_start + self.image_size, x_start:x_start + self.image_size]
            alpha_image = image_data.copy()
            alpha_image[:, :, 3] = image_data[:, :, 3] / overlap_data
            pre_blend_image = np.zeros(self.dest_image_size + (4,), dtype=np.uint8)
            pre_blend_image[y_start:y_start + self.image_size, x_start:x_start + self.image_size] = alpha_image
            pre_blend_images.append(Image.fromarray(pre_blend_image, 'RGBA'))

        dest_image = pre_blend_images[0]
        for blend_image in pre_blend_images[1:]:
            dest_image = Image.alpha_composite(dest_image, blend_image)

        return dest_image

    def interpolate_at_intersection(self, images):

        def interpolate_width(data):
            interpolation_start = data.shape[1] - num_interpolation_pixels
            for i in range(num_interpolation_pixels):
                data[:, interpolation_start + i, 3] *= (num_interpolation_pixels - i) / num_interpolation_pixels
            return data

        def interpolate_height(data):
            interpolation_start = data.shape[0] - num_interpolation_pixels
            for i in range(num_interpolation_pixels):
                data[interpolation_start + i, :, 3] *= (num_interpolation_pixels - i) / num_interpolation_pixels
            return data

        pre_blend_images = []
        num_interpolation_pixels = int(self.image_size * self.interpolation_area)
        for y_idx in range(self.image_rows):
            for x_idx in range(self.image_columns):
                image_idx = y_idx * self.image_columns + x_idx
                image = images[image_idx]
                x_start, y_start = self.get_start_indices(y_idx, x_idx)
                image_data = np.asarray(image).copy().astype(np.float64)

                # create horizontal alpha mask
                if x_idx < self.image_columns - 1:
                    image_data = interpolate_width(image_data)

                if x_idx > 0:
                    image_data = np.fliplr(image_data)
                    image_data = interpolate_width(image_data)
                    image_data = np.fliplr(image_data)

                # create vertical alpha mask
                if y_idx < self.image_rows - 1:
                    image_data = interpolate_height(image_data)

                if y_idx > 0:
                    image_data = np.flipud(image_data)
                    image_data = interpolate_height(image_data)
                    image_data = np.flipud(image_data)

                pre_blend_image = np.zeros(self.dest_image_size + (4,), dtype=np.uint8)
                pre_blend_image[y_start:y_start + self.image_size, x_start:x_start + self.image_size] = image_data.astype(np.uint8)
                pre_blend_images.append(Image.fromarray(pre_blend_image, 'RGBA'))

        dest_image = pre_blend_images[0]
        for blend_image in pre_blend_images[1:]:
            dest_image = Image.alpha_composite(dest_image, blend_image)

        dest_image = dest_image.convert('RGB')
        return dest_image

    def get_start_indices(self, y_idx, x_idx):
        x_start = x_idx * self.image_size
        x_start = x_start - x_idx * self.image_size * self.interpolation_area if x_start > 0 else x_start
        y_start = y_idx * self.image_size
        y_start = y_start - y_idx * self.image_size * self.interpolation_area if y_start > 0 else y_start

        return int(x_start), int(y_start)

    def create_sample(self, image_information):
        images = []
        all_labels = []
        self.overlap_map[:] = 0

        for y_idx in range(self.image_rows):
            x_idx = 0
            while x_idx < self.image_columns:
                image_info = random.choice(image_information)
                file_name = image_info['filename']
                bboxes = image_info['boxes']
                labels = [int(box['label']) - 1 for box in bboxes]
                if len(labels) > self.max_label_length:
                    continue

                all_labels.append(labels + [self.label_pad_value] * (self.max_label_length - len(labels)))

                with Image.open(os.path.join(self.image_dir, file_name)) as image:
                    image = image.resize((self.image_size, self.image_size))
                    image = image.convert('RGBA')
                    images.append(image)

                    x_start, y_start = self.get_start_indices(y_idx, x_idx)
                    self.overlap_map[y_start:y_start + self.image_size, x_start:x_start + self.image_size] += 1
                x_idx += 1

        assert len(images) == self.images_per_sample
        return self.interpolate_at_intersection(images), all_labels

    def pad_labels(self, labels):
        longest_label_length = max(map(len, labels))
        padded_labels = []
        for label in labels:
            padded_label = label + [self.label_pad_value] * (longest_label_length - len(label))
            padded_labels.append(padded_label)

        return padded_labels

    def create_dataset(self, num_samples, image_infos):
        all_labels = []
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(i + 1)

            sample, labels = self.create_sample(image_infos)
            all_labels.extend(labels)

            sample.save(os.path.join(self.destination_image_dir, '{}.png'.format(i)))

        with open(os.path.join(self.destination_dir, '{}.csv'.format(self.dataset_name)), 'w') as gt_file:
            writer = csv.writer(gt_file, delimiter='\t')

            # merge labels per image
            all_labels_concatenated = []
            for idx in range(0, len(all_labels), self.images_per_sample):
                concatenated = list(itertools.chain(*all_labels[idx:idx + self.images_per_sample]))
                all_labels_concatenated.append(concatenated)
            assert len(all_labels_concatenated) == num_samples, "number of labels should be as large as number of generated samples"

            for idx, labels in enumerate(all_labels_concatenated):
                writer.writerow([os.path.join(os.path.abspath(self.destination_image_dir), '{}.png'.format(idx))] + labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that creates a four image svhn dataset for training')
    parser.add_argument('svhn_image_dir', help='path to directory containing svhn images')
    parser.add_argument('svhn_gt', help='path to JSON containing svhn GT')
    parser.add_argument('destination_dir', help='directory where generated samples shall be saved')
    parser.add_argument('num_samples', type=int, help='number of samples to create')
    parser.add_argument('max_label_length', type=int, help='maximum length of labels')
    parser.add_argument('--dataset-name', default='train', help='name of the data set [e.g. train]')
    parser.add_argument('--image-size', type=int, default=100, help='size that each source image shall be resized to')
    parser.add_argument('--image-columns', type=int, default=2, help='number of image columns per generated sample')
    parser.add_argument('--image-rows', type=int, default=2, help='number of image rows per generated sample')

    args = parser.parse_args()

    with open(args.svhn_gt) as gt_json:
        gt_data = json.load(gt_json)

    dataset_creator = SVHNDatasetCreator(
        args.svhn_image_dir,
        args.image_size,
        args.image_columns,
        args.image_rows,
        args.destination_dir,
        args.dataset_name,
        args.max_label_length,
    )
    dataset_creator.create_dataset(args.num_samples, gt_data)
