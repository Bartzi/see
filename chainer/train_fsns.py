import argparse
import copy
import os


import chainer
import datetime

import numpy as np
import shutil
from chainer.training.updaters import MultiprocessParallelUpdater
from chainer.training import extensions

from commands.interactive_train import open_interactive_prompt
from datasets.file_dataset import FileBasedDataset
from datasets.sub_dataset import split_dataset_random, split_dataset, split_dataset_n_random
from insights.bbox_plotter import BBOXPlotter
from insights.fsns_bbox_plotter import FSNSBBOXPlotter
from metrics.ctc_metrics import CTCMetrics
from metrics.lstm_per_step_metrics import PerStepLSTMMetric
from metrics.softmax_metrics import SoftmaxMetrics
from models.fsns import FSNSNet, FSNSSoftmaxRecognitionNet, \
    FSNSSingleSTNLocalizationNet, FSNSSoftmaxRecognitionResNet, FSNSResnetReuseNet
from models.fsns_resnet import FSNSRecognitionResnet
from optimizers.multi_net_optimizer import MultiNetOptimizer
from utils.baby_step_curriculum import BabyStepCurriculum
from utils.datatypes import Size
from utils.intelligent_attribute_shifter import IntelligentAttributeShifter
from utils.multi_accuracy_classifier import Classifier
from utils.train_utils import add_default_arguments, get_fast_evaluator, get_trainer, \
    concat_and_pad_examples, get_definition_filename, get_definition_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to train a text detection network based on Spatial Transformers")
    parser.add_argument('dataset_specification',
                        help='path to json file that contains all datasets to use in a list of dicts')
    parser.add_argument("--blank-label", type=int, help="blank label to use during training")
    parser.add_argument("--char-map", help="path to char map")
    parser.add_argument("--send-bboxes", action='store_true', default=False,
                        help="send predicted bboxes for each iteration")
    parser.add_argument("--port", type=int, default=1337, help="port to connect to for sending bboxes")
    parser.add_argument("--area-factor", type=float, default=0, help="factor for incorporating area loss")
    parser.add_argument("--area-scale-factor", type=float, default=2, help="area scale factor for changing area loss over time")
    parser.add_argument("--aspect-factor", type=float, default=0, help="for for incorporating aspect ratio loss")
    parser.add_argument("--load-localization", action='store_true', default=False, help="only load localization net")
    parser.add_argument("--load-recognition", action='store_true', default=False, help="only load recognition net")
    parser.add_argument("--is-trainer-snapshot", action='store_true', default=False,
                        help="indicate that snapshot to load has been saved by trainer itself")
    parser.add_argument("--no-log", action='store_false', default=True, help="disable logging")
    parser.add_argument("--freeze-localization", action='store_true', default=False,
                        help='freeze weights of localization net')
    parser.add_argument("--zoom", type=float, default=0.9, help="Zoom for initial bias of spatial transformer")
    parser.add_argument("--optimize-all-interval", type=int, default=5,
                        help="interval in which to optimize the whole network instead of only a part")
    parser.add_argument("--use-dropout", action='store_true', default=False, help='use dropout in network')
    parser.add_argument("--test-image", help='path to an image that should be used by BBoxPlotter')
    parser = add_default_arguments(parser)
    args = parser.parse_args()
    args.is_original_fsns = True

    image_size = Size(width=150, height=150)
    target_shape = Size(width=100, height=75)

    # attributes that need to be adjusted, once the Curriculum decides to use
    # a more difficult dataset
    # this is a 'map' of attribute name to a path in the trainer object
    attributes_to_adjust = [
        ('num_timesteps', ['predictor', 'localization_net']),
        ('num_timesteps', ['predictor', 'recognition_net']),
        ('num_timesteps', ['lossfun', '__self__']),
        ('num_labels', ['predictor', 'recognition_net']),
    ]

    # create train curriculum
    curriculum = BabyStepCurriculum(
        args.dataset_specification,
        FileBasedDataset,
        args.blank_label,
        args.gpus,
        attributes_to_adjust=attributes_to_adjust,
        trigger=(args.test_interval, 'iteration'),
        min_delta=0.1,
    )
    train_dataset, validation_dataset = curriculum.load_dataset(0)

    # the metrics object calculates the loss
    metrics = SoftmaxMetrics(
        args.blank_label,
        args.char_map,
        train_dataset.num_timesteps,
        image_size,
        area_loss_factor=args.area_factor,
        aspect_ratio_loss_factor=args.aspect_factor,
        area_scaling_factor=args.area_scale_factor,
        uses_original_data=args.is_original_fsns,
    )

    # create the localization net
    localization_net = FSNSSingleSTNLocalizationNet(
        args.dropout_ratio,
        train_dataset.num_timesteps,
        zoom=args.zoom,
        use_dropout=args.use_dropout,
    )

    # create the recognition net
    recognition_net = FSNSRecognitionResnet(
        target_shape,
        train_dataset.get_label_length(train_dataset.num_timesteps, check_length=False),
        train_dataset.num_timesteps,
        uses_original_data=args.is_original_fsns,
        use_dropout=False,
        dropout_ratio=args.dropout_ratio,
        use_blstm=True,
    )
    net = FSNSNet(localization_net, recognition_net, uses_original_data=args.is_original_fsns)
    model = Classifier(
        net,
        ('accuracy',),
        lossfun=metrics.calc_loss,
        accfun=metrics.calc_accuracy,
        provide_label_during_forward=False
    )

    if args.resume is not None:
        with np.load(args.resume) as f:
            if args.load_localization:
                if args.is_trainer_snapshot:
                    chainer.serializers.NpzDeserializer(f)['/updater/model:main/predictor/localization_net'].load(
                        localization_net)
                else:
                    chainer.serializers.NpzDeserializer(f, strict=False)['localization_net'].load(localization_net)
            elif args.load_recognition:
                if args.is_trainer_snapshot:
                    chainer.serializers.NpzDeserializer(f)['/updater/model:main/predictor/recognition_net'].load(
                        recognition_net
                    )
                else:
                    chainer.serializers.NpzDeserializer(f)['recognition_net'].load(recognition_net)
            else:
                if args.is_trainer_snapshot:
                    chainer.serializers.NpzDeserializer(f)['/updater/model:main/predictor'].load(net)
                else:
                    chainer.serializers.NpzDeserializer(f).load(net)

    base_optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer = base_optimizer
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    optimizer.add_hook(chainer.optimizer.GradientClipping(2))

    # freeze localization net
    if args.freeze_localization:
        localization_net.disable_update()

    # if we are using more than one GPU, we need to evenly split the datasets
    if len(args.gpus) > 1:
        gpu_datasets = split_dataset_n_random(train_dataset, len(args.gpus))
        if not len(gpu_datasets[0]) == len(gpu_datasets[-1]):
            adapted_second_split = split_dataset(gpu_datasets[-1], len(gpu_datasets[0]))[0]
            gpu_datasets[-1] = adapted_second_split
    else:
        gpu_datasets = [train_dataset]

    train_iterators = [chainer.iterators.MultiprocessIterator(dataset, args.batch_size) for dataset in gpu_datasets]
    validation_iterator = chainer.iterators.MultiprocessIterator(validation_dataset, args.batch_size, repeat=False)

    # use the MultiProcessParallelUpdater in order to harness the full power of data parallel computation
    updater = MultiprocessParallelUpdater(train_iterators, optimizer, devices=args.gpus)

    log_dir = os.path.join(args.log_dir, "{}_{}".format(datetime.datetime.now().isoformat(), args.log_name))
    args.log_dir = log_dir

    # backup current file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(__file__, log_dir)

    # backup all necessary configuration params
    report = {
        'log_dir': log_dir,
        'image_size': image_size,
        'target_size': target_shape,
        'localization_net': [localization_net.__class__.__name__, get_definition_filename(localization_net)],
        'recognition_net': [recognition_net.__class__.__name__, get_definition_filename(recognition_net)],
        'fusion_net': [net.__class__.__name__, get_definition_filename(net)],
    }

    for argument in filter(lambda x: not x.startswith('_'), dir(args)):
        report[argument] = getattr(args, argument)

    # callback that logs report (is called by trainer extension that writes the train log)
    def log_postprocess(stats_cpu):
        # only log further information once and not every time we log our progress
        if stats_cpu['epoch'] == 0 and stats_cpu['iteration'] == args.log_interval:
            stats_cpu.update(report)


    # all fields that shall be shown to the user while training
    fields_to_print = [
        'epoch',
        'iteration',
        'main/loss',
        'main/accuracy',
        'lr',
        'fast_validation/main/loss',
        'fast_validation/main/accuracy',
        'validation/main/loss',
        'validation/main/accuracy',
    ]

    # fast evaluator only runs for max. 200 iteration
    FastEvaluator = get_fast_evaluator((args.test_interval, 'iteration'))
    evaluator = (
        FastEvaluator(
            validation_iterator,
            model,
            device=updater._devices[0],
            eval_func=lambda *args: model(*args),
            num_iterations=args.test_iterations,
            converter=concat_and_pad_examples
        ),
        (args.test_interval, 'iteration')
    )
    # epoch validator validates model on complete validation set
    epoch_validation_iterator = copy.copy(validation_iterator)
    epoch_validation_iterator._repeat = False
    epoch_evaluator = (
        chainer.training.extensions.Evaluator(
            epoch_validation_iterator,
            model,
            device=updater._devices[0],
            converter=concat_and_pad_examples,
        ),
        (1, 'epoch')
    )

    model_snapshotter = (
        extensions.snapshot_object(net, 'model_{.updater.iteration}.npz'), (args.snapshot_interval, 'iteration')
    )

    # bbox plotter test
    if not args.test_image:
        test_image = validation_dataset.get_example(0)[0]
    else:
        test_image = train_dataset.load_image(args.test_image)

    # BBOXPlotter performs a forward pass with current state of the model at each iteration, thus enabling
    # the user to inspect the current progress of the model
    bbox_plotter_class = BBOXPlotter if not args.is_original_fsns else FSNSBBOXPlotter
    bbox_plotter = (bbox_plotter_class(
        test_image,
        os.path.join(log_dir, 'boxes'),
        target_shape,
        metrics,
        send_bboxes=args.send_bboxes,
        upstream_port=args.port,
        visualization_anchors=[["localization_net", "vis_anchor"], ["recognition_net", "vis_anchor"]]
    ), (1, 'iteration'))

    # create the trainer object and inject all extensions
    trainer = get_trainer(
        net,
        updater,
        log_dir,
        fields_to_print,
        curriculum=curriculum,
        epochs=args.epochs,
        snapshot_interval=args.snapshot_interval,
        print_interval=args.log_interval,
        extra_extensions=(
            evaluator,
            epoch_evaluator,
            model_snapshotter,
            bbox_plotter,
            (curriculum, (args.test_interval, 'iteration')),
        ),
        postprocess=log_postprocess,
        do_logging=args.no_log,
        model_files=[
            get_definition_filepath(localization_net),
            get_definition_filepath(recognition_net),
            get_definition_filepath(net),
        ]
    )

    # create interactive prompt that can be used to issue commands while the training is in progress
    open_interactive_prompt(
        bbox_plotter=bbox_plotter[0],
        curriculum=curriculum,
    )

    trainer.run()
