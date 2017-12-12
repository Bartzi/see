import argparse
import csv

import copy
import json
import os
import random

from collections import namedtuple

import numpy as np
from PIL import Image
from PIL import ImageStat

Point = namedtuple("Point", ['x', 'y'])


SUPPORTED_IMAGE_TYPES = ['.png']


class BBox:

    def __init__(self, label=0, left=0, width=0, top=0, height=0):
        self.label = [label] if not hasattr(label, '__iter__') else label
        self._left = left
        self._width = width
        self._top = top
        self._height = height

    @property
    def left(self):
        return int(self._left)

    @left.setter
    def left(self, value):
        self._left = int(value)

    @property
    def width(self):
        return int(self._width)

    @width.setter
    def width(self, value):
        self._width = int(value)

    @property
    def top(self):
        return int(self._top)

    @top.setter
    def top(self, value):
        self._top = int(value)

    @property
    def height(self):
        return int(self._height)

    @height.setter
    def height(self, value):
        self._height = int(value)

    def __eq__(self, other):
        return (
            self.left == other.left and
            self.width == other.width and
            self.top == other.top and
            self.height == other.height
        )

    def __str__(self):
        return "l:{}, w:{}, t:{}, h:{}".format(self.left, self.width, self.top, self.height)


def intersects_bbox(point, bbox):
    if bbox.left <= point.x <= bbox.left + bbox.width:
        return True
    elif bbox.top <= point.y <= bbox.top + bbox.height:
        return True
    return False


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left


def intersection(bbox1, bbox2):
    width_overlap = overlap(bbox1.left, bbox1.width, bbox2.left, bbox2.width)
    height_overlap = overlap(bbox1.top, bbox1.height, bbox2.top, bbox2.height)
    if width_overlap < 0 or height_overlap < 0:
        return 0
    return width_overlap * height_overlap


def intersects(bbox1, bbox2):
    return intersection(bbox1, bbox2) > 0


def is_close(questioned, other, epsilon=50):
    return all(b - epsilon <= a <= b + epsilon for a, b in zip(questioned, other))


class SVHNCreator:

    def __init__(self, image_dir, gt, image_size, numbers_per_image):
        self.image_dir = image_dir
        self.gt = gt
        self.image_size = image_size
        self.numbers_per_image = numbers_per_image
        self.max_size_per_number = 0.3

    @staticmethod
    def merge_bboxes(bboxes):
        resulting_bbox = None
        for bbox in bboxes:
            if resulting_bbox is None:
                resulting_bbox = bbox
                continue

            max_right = max(resulting_bbox.left + resulting_bbox.width, bbox.left + bbox.width)
            max_bottom = max(resulting_bbox.top + resulting_bbox.height ,bbox.top + bbox.height)

            resulting_bbox.top = min(resulting_bbox.top, bbox.top)
            resulting_bbox.left = min(resulting_bbox.left, bbox.left)
            resulting_bbox.width = max_right - resulting_bbox.left
            resulting_bbox.height = max_bottom - resulting_bbox.top
            resulting_bbox.label.extend(bbox.label)

        return resulting_bbox

    def create_base_image(self, colour):
        background = Image.new(
            "RGBA",
            (self.image_size, self.image_size),
            colour + (255, ),
        )
        return background

    def fade_image(self, image, fade_percentage=0.4):
        def interpolate_width(data):
            num_interpolation_pixels = int(data.shape[1] * fade_percentage)
            interpolation_start = data.shape[1] - num_interpolation_pixels
            for i in range(num_interpolation_pixels):
                data[:, interpolation_start + i, 3] *= (num_interpolation_pixels - i) / num_interpolation_pixels
            return data

        def interpolate_height(data):
            num_interpolation_pixels = int(data.shape[0] * fade_percentage)
            interpolation_start = data.shape[0] - num_interpolation_pixels
            for i in range(num_interpolation_pixels):
                data[interpolation_start + i, :, 3] *= (num_interpolation_pixels - i) / num_interpolation_pixels
            return data

        image_data = np.asarray(image).copy().astype(np.float64)

        # create horizontal alpha mask
        image_data = interpolate_width(image_data)

        image_data = np.fliplr(image_data)
        image_data = interpolate_width(image_data)
        image_data = np.fliplr(image_data)

        # create vertical alpha mask
        image_data = interpolate_height(image_data)

        image_data = np.flipud(image_data)
        image_data = interpolate_height(image_data)
        image_data = np.flipud(image_data)

        image = Image.fromarray(image_data.astype(np.uint8), mode='RGBA')
        return image

    def crop_number(self, image, bbox, enlarge_percentage=0.5):
        width, height = image.size
        width_enlargement = width * enlarge_percentage
        height_enlargement = height * enlarge_percentage

        larger_bbox = BBox(
            label=bbox.label,
            top=max(bbox.top - height_enlargement // 2, 0),
            left=max(bbox.left - width_enlargement // 2, 0),
            width=bbox.width + width_enlargement,
            height=bbox.height + height_enlargement,
        )

        # larger_bbox = bbox

        cropped_number = image.crop(
            (
                larger_bbox.left,
                larger_bbox.top,
                larger_bbox.left + larger_bbox.width,
                larger_bbox.top + larger_bbox.height,
            )
        )

        cropped_number = self.fade_image(cropped_number)

        return cropped_number, larger_bbox

    def find_paste_location(self, bbox, already_pasted_bboxes):
        original_bbox = copy.copy(bbox)
        while True:
            bbox = original_bbox
            top_left = Point._make((random.randint(0, self.image_size) for _ in range(2)))
            bbox.left = min(top_left.x, self.image_size - bbox.width)
            bbox.top = min(top_left.y, self.image_size - bbox.height)

            if all([intersection(bbox, b) == 0 for b in already_pasted_bboxes]):
                return bbox

    def adjust_house_number_crop(self, crop, bbox):
        max_size = int(self.image_size * self.max_size_per_number)
        if crop.width <= max_size and crop.height <= max_size:
            return crop, bbox

        new_height, new_width = max_size, max_size
        if crop.width < max_size:
            new_width = crop.width
        if crop.height < max_size:
            new_height = crop.height

        crop = crop.resize((new_width, new_height), Image.LANCZOS)
        bbox.width = new_width
        bbox.height = new_height

        return crop, bbox

    def get_dominating_color(self, image):
        image_data = np.asarray(image)
        image_data = np.reshape(image_data, (-1, 4))[:, :-1]
        channel_means = np.mean(image_data, axis=0)
        return [int(mean) for mean in channel_means]

    def get_number_crop(self):
        svhn_image = random.choice(self.gt)
        bboxes = [BBox(
            label=int(b['label']),
            left=b['left'],
            width=b['width'],
            top=b['top'],
            height=b['height']
        ) for b in svhn_image['boxes']]
        merged_bbox = self.merge_bboxes(bboxes)

        with Image.open(os.path.join(self.image_dir, svhn_image['filename'])) as image:
            image = image.convert('RGBA')
            house_number_crop, merged_bbox = self.crop_number(image, merged_bbox)

        dominating_color = self.get_dominating_color(house_number_crop)
        return house_number_crop, merged_bbox, dominating_color

    def sort_bboxes(self, bboxes):
        bboxes = sorted(bboxes, key=lambda x: x.left)
        bboxes = sorted(bboxes, key=lambda x: x.top)
        return bboxes

    def create_sample(self):

        found_bboxes = []
        images = []
        dominating_colour = None
        for _ in range(self.numbers_per_image):
            house_number_crop, bbox, median_colour = self.get_number_crop()
            if len(images) > 0:
                while True:
                    if is_close(median_colour, dominating_colour):
                        break
                    house_number_crop, bbox, median_colour = self.get_number_crop()
            else:
                dominating_colour = median_colour

            house_number_crop, bbox = self.adjust_house_number_crop(house_number_crop, bbox)
            paste_bbox = self.find_paste_location(bbox, found_bboxes)
            found_bboxes.append(paste_bbox)
            images.append(house_number_crop)

        base_image = self.create_base_image(tuple(dominating_colour))

        for image, bbox in zip(images, found_bboxes):
            base_image_data = np.asarray(base_image, dtype=np.uint8).copy()
            image_array = np.asarray(image, dtype=np.uint8)

            image_holder = np.zeros_like(base_image_data, dtype=np.uint8)
            image_holder[bbox.top:bbox.top + bbox.height, bbox.left:bbox.left + bbox.width, :] = image_array[:]
            image = Image.fromarray(image_holder, mode='RGBA')

            base_image_data[bbox.top:bbox.top + bbox.height, bbox.left:bbox.left + bbox.width, 3] = 255 - image_array[..., 3]
            base_image = Image.fromarray(base_image_data, mode='RGBA')
            base_image = Image.alpha_composite(base_image, image)

        return base_image, found_bboxes

    def create_dataset(self, num_samples, destination_dir, max_label_length, pad_value=10, start=0):
        i = start
        label_file_data = []
        bbox_file_data = []
        while i < num_samples + start:
            image, bboxes = self.create_sample()
            image = image.convert('RGB')
            labels = [bbox.label for bbox in self.sort_bboxes(bboxes)]

            if not all(len(label) <= max_label_length for label in labels):
                continue

            labels = [l + [pad_value] * (max_label_length - len(l)) for l in labels]
            labels_concatentated = []
            for label in labels:
                labels_concatentated.extend(label)
            labels = labels_concatentated

            image_path = os.path.join(os.path.abspath(destination_dir), "{}.png".format(i))
            image.save(image_path)

            bbox_data = []
            for bbox in bboxes:
                bbox_data.extend([
                    bbox.left,
                    bbox.width,
                    bbox.top,
                    bbox.height,
                ])
            bbox_file_data.append([image_path] + bbox_data)
            label_file_data.append([image_path] + labels + bbox_data)

            if (i + 1) % 1000 == 0:
                print(i + 1)
            i += 1

        with open(os.path.join(os.path.dirname(destination_dir), "{}.csv".format(os.path.basename(destination_dir))), 'w' if start == 0 else 'a') as gt_file:
            writer = csv.writer(gt_file, delimiter='\t')
            writer.writerows(label_file_data)

        with open(os.path.join(os.path.dirname(destination_dir), "{}_bboxes.csv".format(os.path.basename(destination_dir))), 'w' if start == 0 else 'a') as gt_file:
            writer = csv.writer(gt_file, delimiter='\t')
            writer.writerows(bbox_file_data)


class GaussianSVHNCreator(SVHNCreator):

    def __init__(self, *args, **kwargs):
        self.variance = kwargs.pop('variance')

        super(GaussianSVHNCreator, self).__init__(*args, **kwargs)

    def find_paste_location(self, bbox, already_pasted_bboxes):

        while True:
            x_derivation = random.gauss(0, self.variance) * (self.image_size // 2)
            y_derivation = random.gauss(0, self.variance) * (self.image_size // 2)
            center = Point(x=self.image_size // 2, y=self.image_size // 2)

            bbox.left = max(min(center.x + x_derivation, self.image_size), 0)
            bbox.top = max(min(center.y + y_derivation, self.image_size), 0)

            if bbox.left + bbox.width > self.image_size:
                bbox.left = self.image_size - bbox.width
            if bbox.top + bbox.height > self.image_size:
                bbox.top = self.image_size - bbox.height

            if not any(intersects(bbox, box) for box in already_pasted_bboxes):
                return bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that creates multiple svhn image dataset for training')
    parser.add_argument('svhn_image_dir', help='path to directory containing svhn images')
    parser.add_argument('svhn_gt', help='path to JSON containing svhn GT')
    parser.add_argument('destination_dir', help='directory where generated samples shall be saved')
    parser.add_argument('num_samples', type=int, help='number of samples to create')
    parser.add_argument('max_label_length', type=int, help='maximum length of labels')
    parser.add_argument('--dataset-name', default='train', help='name of the data set [e.g. train]')
    parser.add_argument('--image-size', type=int, default=200, help='size that each source image shall be resized to')
    parser.add_argument('--numbers_per_image', type=int, default=4, help='number of house numbers per image')
    parser.add_argument('--variance', type=float, default=1.0, help='variance for gaussian distribution that places individual numbers')
    parser.add_argument('--enlarge', action='store_true', default=False, help='indicates that you want to enlarge the already existing dataset')

    args = parser.parse_args()

    with open(args.svhn_gt) as gt_file:
        svhn_gt = json.load(gt_file)

    destination_dir = os.path.join(args.destination_dir, args.dataset_name)
    os.makedirs(destination_dir, exist_ok=True)

    if args.enlarge:
        num_files = len(list(filter(lambda x: os.path.splitext(x)[-1] in SUPPORTED_IMAGE_TYPES, os.listdir(destination_dir))))
        start = num_files
    else:
        start = 0

    dataset_creator = GaussianSVHNCreator(
        args.svhn_image_dir,
        svhn_gt,
        args.image_size,
        args.numbers_per_image,
        variance=args.variance,
    )
    dataset_creator.create_dataset(args.num_samples, destination_dir, args.max_label_length, start=start)
