import argparse
import os
import re

from collections import namedtuple

from PIL import Image
from PIL import ImageChops
from PIL.GifImagePlugin import getheader, getdata

SUPPORTED_IMAGETYPES = [".png", ".jpg", ".jpeg"]
ImageData = namedtuple("ImageData", ["file_name", "image"])


def intToBin(i):
    """ Integer to two bytes """
    # devide in two parts (bytes)
    i1 = i % 256
    i2 = int( i/256)
    # make string (little endian)
    return chr(i1) + chr(i2)


def create_loop_header(loops=0):
    if loops == 0 or loops == float('inf'):
        loops = 2 ** 16 - 1

    bb = "\x21\xFF\x0B"  # application extension
    bb += "NETSCAPE2.0"
    bb += "\x03\x01"
    bb += intToBin(loops)
    bb += '\x00'  # end
    return [bb.encode('utf-8')]


def makedelta(fp, sequence):
    """Convert list of image frames to a GIF animation file"""

    frames = 0

    previous = None

    for im in sequence:

        # To specify duration, add the time in milliseconds to getdata(),
        # e.g. getdata(im, duration=1000)

        if not previous:

            # global header
            loops = 2 ** 16 - 1
            for s in getheader(im, info={"loop": loops})[0] + getdata(im, duration=10, loop=2 ** 16 - 1):
                fp.write(s)

        else:

            # delta frame
            delta = ImageChops.subtract_modulo(im, previous)

            bbox = delta.getbbox()

            if bbox:

                # compress difference
                for s in getdata(im.crop(bbox), offset=bbox[:2], duration=10):
                    fp.write(s)

            else:
                # FIXME: what should we do in this case?
                pass

        previous = im.copy()

        frames += 1

    fp.write(b";")

    return frames


def make_gif(image_dir, dest_file, pattern="(\d+)"):
    sort_pattern = re.compile(pattern)

    image_files = filter(lambda x: os.path.splitext(x)[-1] in SUPPORTED_IMAGETYPES, os.listdir(image_dir))
    images = []

    try:
        print("loading images")
        for file_name in image_files:
            path = os.path.join(image_dir, file_name)
            images.append(ImageData._make((file_name, Image.open(path).convert('P'))))

        print("sorting images")
        images_sorted = sorted(images, key=lambda x: int(re.search(sort_pattern, x.file_name).group(1)))

        print("writing gif")
        with open(dest_file, "wb") as out_file:
            makedelta(out_file, [image.image for image in images_sorted])

    finally:
        for image in images:
            image.image.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that creates a gif out of a number of given input images')
    parser.add_argument("image_dir", help="path to directory that contains all images that shall be converted to a gif")
    parser.add_argument("dest_file", help="path to destination gif file")
    parser.add_argument("--pattern", default="(\d+)", help="naming pattern to extract the ordering of the images")

    args = parser.parse_args()

    make_gif(args.image_dir, args.dest_file, args.pattern)