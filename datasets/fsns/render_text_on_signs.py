import argparse
import csv
import json

import random

import os
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from functools import lru_cache


SUPPORTED_IMAGES = ['.jpg', '.png', '.jpeg']
FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fonts'))
FONTS = [os.path.abspath(os.path.join(FONT_DIR, font)) for font in os.listdir(FONT_DIR)]
MAX_IMAGES_PER_DIR = 1000


def get_image_paths(dir):
    all_images = []
    for root, _, files in os.walk(dir):
        files = [os.path.join(root, f) for f in filter(lambda x: os.path.splitext(x)[-1].lower() in SUPPORTED_IMAGES, files)]
        all_images.extend(files)

    return all_images


@lru_cache(maxsize=1024)
def get_image(image_path):
    return Image.open(image_path)


@lru_cache(maxsize=1024)
def find_font_size(draw, text_lines, max_width, max_height, spacing):
    # start with a default font size that should be large enough to be too large
    font_size = 35

    # reload the font until the word fits or the font size would be too small
    while True:
        font = ImageFont.truetype(random.choice(FONTS), size=font_size, encoding='unic')
        text_width, text_height = draw.multiline_textsize(text_lines, font, spacing=spacing)

        if text_width <= max_width and text_height <= max_height:
            return font, (text_width, text_height)

        font_size -= 1

        if font_size <= 1:
            raise ValueError('Can not draw Text on given image')


def random_crop(image, crop_width, crop_height):
    left = min(random.choice(range(image.width)), image.width - crop_width)
    top = min(random.choice(range(image.height)), image.height - crop_height)

    return image.crop((
        left,
        top,
        left + crop_width,
        top + crop_height,
    ))


def save_image(index, image, base_dest_dir, split_ratio):

    def get_subdir():
        return os.path.join(
            "{:04d}".format(index // (MAX_IMAGES_PER_DIR * MAX_IMAGES_PER_DIR)),
            "{:04d}".format(index // MAX_IMAGES_PER_DIR)
        )

    split_dir = "validation" if random.randint(1, 100) < split_ratio else "train"
    dest_dir = os.path.join(base_dest_dir, split_dir, get_subdir())
    os.makedirs(dest_dir, exist_ok=True)

    image_path = "{}.png".format(os.path.join(dest_dir, str(index)))
    image.save(image_path)

    return image_path, split_dir == 'train'


def get_labels(words, char_map, blank_label, max_length, max_textlines):
    all_labels = []

    for word in words:
        labels = [char_map[ord(char)] for char in word]
        labels.extend([blank_label] * (max_length - len(labels)))
        all_labels.append(labels)

    if len(all_labels) < max_textlines:
        all_labels.extend([[blank_label] * max_length for _ in range(max_textlines - len(all_labels))])

    return [label for labels in all_labels for label in labels]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that renders text on street sign images and puts them into a nice context')
    parser.add_argument('wordlist', help='path to wordlist file')
    parser.add_argument('sign_dir', help='path to a directory containing empty signs to render text on')
    parser.add_argument('background_dir', help='path to a directory holding dirs/images of background images where signs will be placed on')
    parser.add_argument('destination', help='where to put the generated samples')
    parser.add_argument('char_map', help='path to json file that contains a char map')
    parser.add_argument('num_samples', type=int, help='number of samples to create')
    parser.add_argument('--blank-label', type=int, default=133, help='index of blank label [default: 133]')
    parser.add_argument('--max-length', type=int, default=10, help='max length of word on signs [default: 10]')
    parser.add_argument('--label-length', type=int, default=37, help='label length for each rendered text line [default: 37]')
    parser.add_argument('--image-size', type=int, default=150, help='size of resulting images [default: 150 x 150]')
    parser.add_argument('--split-ratio', type=int, default=15, help='percentage of samples to use for validation [default: 15]')
    parser.add_argument('--max-textlines', type=int, default=3, help='maximum number of text lines per rendered image [default: 3]')
    parser.add_argument('--min-textlines', type=int, default=1, help='minimum number of text lines per rendered image [default: 1]')

    args = parser.parse_args()

    print("loading sign images")
    sign_image_paths = get_image_paths(args.sign_dir)
    sign_images = [get_image(path) for path in sign_image_paths]

    print("getting background images")
    background_image_paths = get_image_paths(args.background_dir)

    print("loading wordlist")
    with open(args.wordlist) as the_wordlist:
        wordlist = [word.strip() for word in filter(lambda x: len(x) <= args.max_length and "'" not in x, the_wordlist)]

    print("opening char map")
    with open(args.char_map) as the_map:
        char_map = json.load(the_map)
        reverse_char_map = {v: k for k, v in char_map.items()}

    print("starting generation of samples")
    os.makedirs(args.destination, exist_ok=True)
    with open(os.path.join(args.destination, 'train.csv'), 'w') as train_labels, open(os.path.join(args.destination, 'val.csv'), 'w') as val_labels:
        train_writer = csv.writer(train_labels, delimiter='\t')
        val_writer = csv.writer(val_labels, delimiter='\t')

        i = 0
        start_time = time.time()
        while i <= args.num_samples:
            num_textlines = random.randint(args.min_textlines, args.max_textlines)
            words = [random.choice(wordlist) for _ in range(num_textlines)]

            sign_image = random.choice(sign_images).copy()
            draw = ImageDraw.Draw(sign_image)

            background_image = get_image(random.choice(background_image_paths))
            background_image = random_crop(background_image, args.image_size, args.image_size)
            background_image = background_image.convert('RGBA')

            width, height = sign_image.size
            max_width = width * 0.7
            max_height = height * 0.7
            spacing = 5

            text_lines = '\n'.join(words)
            try:
                font, text_size = find_font_size(draw, text_lines, max_width, max_height, spacing)
            except ValueError:
                continue

            draw.multiline_text(
                (width // 2 - text_size[0] // 2, height // 2 - text_size[1] // 2),
                text_lines,
                fill='white',
                font=font,
                spacing=spacing,
                align='center',
            )

            paste_box = (
                args.image_size // 2 - sign_image.width // 2,
                args.image_size // 2 - sign_image.height // 2,
            )

            overlay_image = Image.new("RGBA", (args.image_size, args.image_size), (255, 255, 255, 0))
            overlay_image.paste(sign_image, box=paste_box)

            background_image = Image.alpha_composite(background_image, overlay_image)

            sample_path, is_train = save_image(i, background_image, args.destination, args.split_ratio)
            labels = get_labels(words, reverse_char_map, args.blank_label, args.label_length, args.max_textlines)

            if is_train:
                train_writer.writerow((sample_path, *labels))
            else:
                val_writer.writerow((sample_path, *labels))

            i += 1
            if i % 1000 == 0:
                print("Generated {} samples, took {:4.5} seconds".format(i, time.time() - start_time))
                start_time = time.time()
