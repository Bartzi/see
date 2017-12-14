import argparse
import os
import re

from collections import namedtuple

import subprocess

import shutil
import tempfile


SUPPORTED_IMAGETYPES = [".png", ".jpg", ".jpeg"]
ImageData = namedtuple('ImageData', ['file_name', 'path'])


def get_filter(pattern):
    def filter_function(x):
        return int(re.search(pattern, x.file_name).group(1))
    return filter_function


def make_video(image_dir, dest_file, batch_size=1000, start=None, end=None, pattern=r"(\d+)"):
    sort_pattern = re.compile(pattern)

    image_files = os.listdir(image_dir)

    image_files = filter(lambda x: os.path.splitext(x)[-1] in SUPPORTED_IMAGETYPES, image_files)
    images = []

    print("loading images")
    for file_name in image_files:
        path = os.path.join(image_dir, file_name)
        images.append(ImageData(file_name=file_name, path=path))

    extract_number = get_filter(sort_pattern)
    if end is None:
        end = extract_number(max(images, key=extract_number))
    if start is None:
        start = 0

    print("sort and cut images")
    images_sorted = list(filter(
        lambda x: start <= extract_number(x) < end,
        sorted(images, key=extract_number)))

    print("creating temp file")
    temp_file = tempfile.NamedTemporaryFile(mode="w")
    video_dir = tempfile.mkdtemp()
    i = 1
    try:
        # create a bunch of videos and merge them later (saves memory)
        while i < len(images_sorted):
            image = images_sorted[i]
            if i % batch_size == 0:
                temp_file = create_video(i, temp_file, video_dir)
            else:
                print(image.path, file=temp_file)
            i += 1

        if i % batch_size != 0:
            print("creating last video")
            temp_file = create_video(i - 1, temp_file, video_dir)
        temp_file.close()

        # merge created videos
        process_args = [
            'ffmpeg',
            '-i concat:"{}"'.format(
                '|'.join(sorted(
                    os.listdir(video_dir),
                    key=lambda x: int(os.path.splitext(x.rsplit('/', 1)[-1])[0]))
                )
            ),
            '-c copy {}'.format(os.path.abspath(dest_file))
        ]
        print(' '.join(process_args))
        subprocess.run(' '.join(process_args), shell=True, check=True, cwd=video_dir)
    finally:
        shutil.rmtree(video_dir)


def create_video(i, temp_file, video_dir):
    process_args = [
        'convert',
        '-quality 100',
        '@{}'.format(temp_file.name),
        os.path.join(video_dir, "{}.mpeg".format(i))
    ]
    print(' '.join(process_args))
    temp_file.flush()
    subprocess.run(' '.join(process_args), shell=True, check=True)
    temp_file.close()
    temp_file = tempfile.NamedTemporaryFile(mode="w")
    return temp_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that creates a gif out of a number of given input images')
    parser.add_argument("image_dir", help="path to directory that contains all images that shall be converted to a gif")
    parser.add_argument("dest_file", help="path to destination gif file")
    parser.add_argument("--pattern", default=r"(\d+)", help="naming pattern to extract the ordering of the images")
    parser.add_argument("--batch-size", "-b", default=1000, type=int, help="batch size for processing, [default=1000]")
    parser.add_argument("-e", "--end", type=int, help="maximum number of images to put in gif")
    parser.add_argument("-s", "--start", type=int, help="frame to start")

    args = parser.parse_args()

    make_video(args.image_dir, args.dest_file, batch_size=args.batch_size, start=args.start, end=args.end, pattern=args.pattern)
