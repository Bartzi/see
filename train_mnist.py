import argparse
import os
import statistics
from itertools import zip_longest

import chainer
import datetime

import numpy as np
from PIL import Image
from chainer import cuda
import chainer.functions as F
from chainer.training import extensions

from datasets.mnist_dataset import MNISTDataset, HDF5MnistDataset, FileBasedMNISTDataset
from insights.bbox_plotter import BBOXPlotter
from models.mnist import MNISTNet, MNISTLocalizationNet, MNISTRecognitionNet
from models.text_detection_net import SmallLocalizationNet, TextDetectionNet
from utils.datatypes import Size
from utils.multi_accuracy_classifier import Classifier
from utils.train_utils import add_default_arguments, get_fast_evaluator, AttributeUpdater, get_trainer, \
    concat_and_pad_examples, TwoStateLearningRateShifter


def mnist_loss(x, t):
    xp = cuda.get_array_module(x[0].data, t.data)
    batch_predictions, _, _ = x
    losses = []

    for predictions, labels in zip(F.split_axis(batch_predictions, args.timesteps, axis=1), F.separate(t, axis=1)):
        batch_size, _, num_classes = predictions.data.shape
        predictions = F.reshape(F.flatten(predictions), (batch_size, num_classes))
        losses.append(F.softmax_cross_entropy(predictions, labels))

    return sum(losses)


def mnist_accuracy(x, t):
    xp = cuda.get_array_module(x[0].data, t.data)
    batch_predictions, _, _ = x
    accuracies = []

    for predictions, labels in zip(F.split_axis(batch_predictions, args.timesteps, axis=1), F.separate(t, axis=1)):
        batch_size, _, num_classes = predictions.data.shape
        predictions = F.reshape(F.flatten(predictions), (batch_size, num_classes))
        accuracies.append(F.accuracy(predictions, labels))

    return sum(accuracies) / max(len(accuracies), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to train a text detection network based on Spatial Transformers")
    parser.add_argument("train_data", help="path to train data file")
    parser.add_argument("val_data", help="path to validation data file")
    parser.add_argument("--timesteps", type=int, default=5, help="number of timesteps the GRU shall run [default: 5]")
    parser.add_argument("--alternative", action="store_true", default=False, help="use alternative implementation of spatial Transformers")
    parser.add_argument("-ds", dest='downsample_factor', type=int, default=2, help="downsample for image sampler")
    parser = add_default_arguments(parser)
    args = parser.parse_args()

    image_size = Size(width=200, height=200)

    localization_net = MNISTLocalizationNet(args.dropout_ratio, args.timesteps)
    recognition_net = MNISTRecognitionNet(image_size, args.dropout_ratio, downsample_factor=args.downsample_factor, use_alternative=args.alternative)
    net = MNISTNet(localization_net, recognition_net)

    model = Classifier(net, ('accuracy', ), lossfun=mnist_loss, accfun=mnist_accuracy)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer = chainer.optimizers.MomentumSGD(lr=args.learning_rate)
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    # lr_shifter = AttributeUpdater(0.1, trigger=(5, 'epoch'))
    # optimizer = chainer.optimizers.RMSprop(lr=args.learning_rate)
    # optimizer = chainer.optimizers.AdaDelta(rho=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    train_dataset = FileBasedMNISTDataset(args.train_data)
    validation_dataset = FileBasedMNISTDataset(args.val_data)

    # train_dataset = MNISTDataset(args.train_data, "train")
    # validation_dataset = MNISTDataset(args.train_data, "valid")

    train_iterator = chainer.iterators.MultiprocessIterator(train_dataset, args.batch_size)
    validation_iterator = chainer.iterators.MultiprocessIterator(validation_dataset, args.batch_size)

    updater = chainer.training.StandardUpdater(train_iterator, optimizer, device=args.gpu)

    log_dir = os.path.join(args.log_dir, "{}_{}".format(args.log_name, datetime.datetime.now().isoformat()))

    fields_to_print = [
        'epoch',
        'iteration',
        'main/loss',
        'main/accuracy',
        'validation/main/loss',
        'validation/main/accuracy',
    ]

    FastEvaluator = get_fast_evaluator((args.test_interval, 'iteration'))
    evaluator = FastEvaluator(validation_iterator, model, device=args.gpu, eval_func=lambda *args: model(*args),
                              num_iterations=args.test_iterations, converter=concat_and_pad_examples)

    # take snapshot of model every 5 epochs
    model_snapshotter = extensions.snapshot_object(net, 'model_{.updater.iteration}.npz', trigger=(5, 'epoch'))

    # bbox plotter test
    test_image = validation_dataset.get_example(0)[0]
    bbox_plotter = BBOXPlotter(test_image, os.path.join(log_dir, 'boxes'), args.downsample_factor, send_bboxes=True)

    learning_rate_schedule = [
        {
            "state": TwoStateLearningRateShifter.INTERVAL_BASED_SHIFT_STATE,
            "target_lr": 5e-4,
            "update_trigger": (10, 'epoch'),
            "stop_trigger": (70, 'epoch'),
        },
        {
            "state": TwoStateLearningRateShifter.CONTINUOS_SHIFT_STATE,
            "target_lr": 5e-10,
            "update_trigger": (15, 'epoch'),
            "stop_trigger": (90, 'epoch'),
        },
    ]

    # num_epochs = sum([phase["stop_trigger"][0] for phase in learning_rate_schedule])
    #
    # lr_shifter = TwoStateLearningRateShifter(args.learning_rate, learning_rate_schedule)

    trainer = get_trainer(
        net,
        updater,
        log_dir,
        fields_to_print,
        epochs=args.epochs,
        snapshot_interval=args.snapshot_interval,
        print_interval=args.log_interval,
        extra_extensions=(
            evaluator,
            model_snapshotter,
            bbox_plotter,
            # lr_shifter,
        )
    )

    if args.resume is not None:
        print("resuming training from {}".format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
