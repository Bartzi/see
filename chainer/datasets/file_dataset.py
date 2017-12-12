import csv
import json
import os

try:
    import cv2
except ImportError:
    pass

import numpy as np
import random
from PIL import Image

from chainer.dataset import dataset_mixin


class FileBasedDataset(dataset_mixin.DatasetMixin):

    def __init__(self, dataset_file, file_contains_metadata=True, resize_size=None):
        self.file_names = []
        self.labels = []
        self.base_dir = os.path.dirname(dataset_file)
        self.num_timesteps = None
        self.num_labels = None
        self.resize_size = resize_size

        with open(dataset_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            if file_contains_metadata:
                # first read metadata about file
                self.num_timesteps, self.num_labels = (int(i) for i in next(reader))
            # then read all data
            for line in reader:
                file_name = line[0]
                labels = np.array(line[1:], dtype=np.int32)
                self.file_names.append(file_name)
                self.labels.append(labels)

        assert len(self.file_names) == len(self.labels)
        label_length = len(self.labels[0])
        for i, label in enumerate(self.labels):
            if len(label) != label_length:
                print("Label of file {} is not as long as all others ({} vs {})".format(self.file_names[i], len(label), label_length))

    def __len__(self):
        return len(self.file_names)

    def pad_labels(self, new_num_timesteps, pad_value):
        padded_labels = [
            np.concatenate(
                (label, np.array([pad_value] * (new_num_timesteps - self.num_timesteps) * self.num_labels, dtype=np.int32)),
                axis=0
            ) for label in self.labels
        ]
        self.num_timesteps = new_num_timesteps
        self.labels = padded_labels

    def get_label_length(self, num_timesteps, check_length=True):
        label_length, rest = divmod(len(self.labels[0]), num_timesteps)
        if check_length:
            assert rest == 0, "Number of labels does not evenly divide by number of timesteps! (Rest: {})".format(rest)
        return label_length

    def load_image(self, file_name):
        with Image.open(os.path.join(self.base_dir, file_name)) as the_image:
            the_image = the_image.convert("RGB")
            if self.resize_size is not None:
                the_image = the_image.resize((self.resize_size.width, self.resize_size.height), Image.LANCZOS)
                assert the_image.width == self.resize_size.width
                assert the_image.height == self.resize_size.height
                assert the_image.mode == 'RGB'

            image = np.asarray(the_image, dtype=np.float32)
            image /= 255

        # put color channels to the front, as expected by Chainer
        image = image.transpose(2, 0, 1)
        num_channels, height, width = image.shape
        assert num_channels == 3
        if self.resize_size is not None:
            assert height == self.resize_size.height
            assert width == self.resize_size.width

        return image

    def get_example(self, i):
        while True:
            try:
                image = self.load_image(self.file_names[i])
                break
            except Exception as e:
                print("could not load image: {}".format(self.file_names[i]))
                i = random.randint(0, len(self))

        label = self.labels[i]
        return image, label


class TextRecFileDataset(dataset_mixin.DatasetMixin):

    def __init__(self, dataset_file, char_map=None, file_contains_metadata=True, resize_size=None, blank_label=0):
        self.file_names = []
        self.labels = []
        self.base_dir = os.path.dirname(dataset_file)
        self.num_timesteps = None
        self.num_labels = None
        self.resize_size = resize_size
        self.blank_label = blank_label

        with open(char_map) as the_map:
            self.char_map = json.load(the_map)
            self.reverse_char_map = {v: k for k, v in self.char_map.items()}

        with open(dataset_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            if file_contains_metadata:
                # first read metadata about file
                self.num_timesteps, self.num_labels = (int(i) for i in next(reader))
            # then read all data
            for line in reader:
                file_name = line[0]
                labels = line[1]
                self.file_names.append(file_name)
                self.labels.append(labels)

        assert len(self.file_names) == len(self.labels)

    def __len__(self):
        return len(self.file_names)

    def get_label_length(self, num_timesteps, check_length=True):
        return self.num_labels

    def get_example(self, i):
        try:
            image = self.load_image(self.file_names[i])
        except Exception as e:
            print("can not load image: {}".format(self.file_names[i]))
            i = random.randint(0, len(self))
            image = self.load_image(self.file_names[i])

        labels = self.get_labels(self.labels[i])
        return image, labels

    def load_image(self, file_name):
        with Image.open(os.path.join(self.base_dir, file_name)) as the_image:
            the_image = the_image.convert('RGB')
            if self.resize_size is not None:
                the_image = the_image.resize((self.resize_size.width, self.resize_size.height), Image.LANCZOS)
            image = np.asarray(the_image, dtype=np.float32)
            image /= 255
        del the_image

        image = image.transpose(2, 0, 1)
        return image

    def get_labels(self, word):
        labels = [int(self.reverse_char_map[ord(character)]) for character in word]
        labels += [self.blank_label] * (self.num_timesteps - len(labels))
        return np.array(labels, dtype=np.int32)


class OpencvTextRecFileDataset(TextRecFileDataset):

    def load_image(self, file_name):
        the_image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        the_image = cv2.cvtColor(the_image, cv2.COLOR_BGR2RGB)
        if self.resize_size is not None:
            the_image = cv2.resize(the_image, self.resize_size)

        the_image = the_image.astype(np.float32) / 255
        image = np.transpose(the_image, (2, 0, 1))
        del the_image

        return image

