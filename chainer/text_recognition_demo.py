import argparse
import importlib

import os

import json
from collections import OrderedDict

import chainer
from pprint import pprint

import chainer.functions as F
import numpy as np

from PIL import Image
from chainer import configuration

from utils.datatypes import Size


def get_class_and_module(log_data):
    if not isinstance(log_data, list):
        if 'InverseCompositional' in log_data:
            module_name = 'ic_stn.py'
            klass_name = log_data
        else:
            module_name = 'text_recognition.py'
            klass_name = log_data
    else:
        klass_name, module_name = log_data
    return klass_name, module_name


def load_module(module_file):
    module_spec = importlib.util.spec_from_file_location("models.model", module_file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_recognition_net(recognition_net_class, target_shape, args):
    return recognition_net_class(
        target_shape,
        num_rois=args.timesteps,
        label_size=52,
    )


def build_localization_net(localization_net_class, target_shape, args):
    return localization_net_class(
        args.dropout_ratio,
        args.timesteps,
        0,
        target_shape,
        zoom=1.0,
        do_parameter_refinement=False
    )


def build_fusion_net(fusion_net_class, localization_net, recognition_net):
    return fusion_net_class(localization_net, recognition_net)


def create_network(args, log_data):
    # Step 1: build network
    localization_net_class_name, localization_module_name = get_class_and_module(log_data['localization_net'])
    module = load_module(os.path.abspath(os.path.join(args.model_dir, localization_module_name)))
    localization_net_class = eval('module.{}'.format(localization_net_class_name))
    localization_net = build_localization_net(localization_net_class, log_data['target_size'], args)

    recognition_net_class_name, recognition_module_name = get_class_and_module(log_data['recognition_net'])
    module = load_module(os.path.abspath(os.path.join(args.model_dir, recognition_module_name)))
    recognition_net_class = eval('module.{}'.format(recognition_net_class_name))
    recognition_net = build_recognition_net(recognition_net_class, target_shape, args)

    fusion_net_class_name, fusion_module_name = get_class_and_module(log_data['fusion_net'])
    module = load_module(os.path.abspath(os.path.join(args.model_dir, fusion_module_name)))
    fusion_net_class = eval('module.{}'.format(fusion_net_class_name))
    net = build_fusion_net(fusion_net_class, localization_net, recognition_net)

    if args.gpu >= 0:
        net.to_gpu(args.gpu)

    return net


def load_image(image_file, xp, image_size):
    with Image.open(image_file) as the_image:
        the_image = the_image.convert('L')
        the_image = the_image.resize((image_size.width, image_size.height), Image.LANCZOS)
        image = xp.asarray(the_image, dtype=np.float32)
        image /= 255
        image = xp.broadcast_to(image, (3, image_size.height, image_size.width))
        return image


def strip_prediction(predictions, xp, blank_symbol):
    words = []
    for prediction in predictions:
        blank_symbol_seen = False
        stripped_prediction = xp.full((1,), prediction[0], dtype=xp.int32)
        for char in prediction:
            if char == blank_symbol:
                blank_symbol_seen = True
                continue
            if char == stripped_prediction[-1] and not blank_symbol_seen:
                continue
            blank_symbol_seen = False
            stripped_prediction = xp.hstack((stripped_prediction, char.reshape(1, )))
        words.append(stripped_prediction)
    return words


def extract_bbox(bbox, image_size, target_shape, xp):
    bbox.data[...] = (bbox.data[...] + 1) / 2
    bbox.data[0, :] *= image_size.width
    bbox.data[1, :] *= image_size.height

    x = xp.clip(bbox.data[0, :].reshape(target_shape), 0, image_size.width)
    y = xp.clip(bbox.data[1, :].reshape(target_shape), 0, image_size.height)

    top_left = (float(x[0, 0]), float(y[0, 0]))
    bottom_right = (float(x[-1, -1]), float(y[-1, -1]))

    return top_left, bottom_right


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that loads model and predicts on a given image")
    parser.add_argument("model_dir", help="path to directory where model is saved")
    parser.add_argument("snapshot_name", help="name of the snapshot to load")
    parser.add_argument("image_path", help="path to the image that shall be evaluated")
    parser.add_argument("char_map", help="path to char map, that maps class id to character")
    parser.add_argument("--gpu", type=int, default=-1, help="id of gpu to use [default: use cpu]")

    args = parser.parse_args()
    # set standard args that should always hold true if using the supplied model
    args.log_name = 'log'
    args.dropout_ratio = 0.5
    args.blank_symbol = 0
    # max number of text regions in the image
    args.timesteps = 23
    # max number of characters per word
    args.num_labels = 1

    # open log and extract meta information
    with open(os.path.join(args.model_dir, args.log_name)) as the_log:
        log_data = json.load(the_log)[0]

    target_shape = Size._make(log_data['target_size'])
    image_size = Size._make(log_data['image_size'])

    xp = chainer.cuda.cupy if args.gpu >= 0 else np
    network = create_network(args, log_data)

    # load weights
    with np.load(os.path.join(args.model_dir, args.snapshot_name)) as f:
        chainer.serializers.NpzDeserializer(f).load(network)

    # load char map
    with open(args.char_map) as the_map:
        char_map = json.load(the_map)

    # load image
    image = load_image(args.image_path, xp, image_size)
    with configuration.using_config('train', False):
        predictions, crops, grids = network(image[xp.newaxis, ...])

    # extract class scores for each word
    words = OrderedDict({})

    predictions = F.concat([F.expand_dims(prediction, axis=0) for prediction in predictions], axis=0)

    classification = F.softmax(predictions, axis=2)
    classification = classification.data
    classification = xp.argmax(classification, axis=2)
    classification = xp.transpose(classification, (1, 0))

    word = strip_prediction(classification, xp, args.blank_symbol)[0]

    word = "".join(map(lambda x: chr(char_map[str(x)]), word))

    bboxes = []
    for bbox in grids[0]:
        bbox = extract_bbox(bbox, image_size, target_shape, xp)
        bboxes.append(OrderedDict({
            'top_left': bbox[0],
            'bottom_right': bbox[1]
        }))
    words[word] = bboxes

    pprint(words)




