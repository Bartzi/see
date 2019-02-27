import argparse
import copy
import datetime
import json
import os
import shutil

import chainer
import numpy as np
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions
from chainer.training.updaters import MultiprocessParallelUpdater

from commands.interactive_train import open_interactive_prompt
from datasets.file_dataset import TextRecFileDataset
from datasets.sub_dataset import split_dataset, split_dataset_n_random
from insights.text_rec_bbox_plotter import TextRecBBOXPlotter
from metrics.textrec_metrics import TextRecSoftmaxMetrics
from models.ic_stn import InverseCompositionalLocalizationNet
from models.text_recognition import TextRecognitionNet, TextRecNet
from utils.baby_step_curriculum import BabyStepCurriculum
from utils.datatypes import Size
from utils.multi_accuracy_classifier import Classifier
from utils.train_utils import add_default_arguments, get_fast_evaluator, get_trainer, \
    get_concat_and_pad_examples, get_definition_filepath, get_definition_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to train a text detection network based on Spatial Transformers")
    parser.add_argument('dataset_specification',
                        help='path to json file that contains all datasets to use in a list of dicts')
    parser.add_argument("--blank-label", type=int, default=0, help="blank label to use during training")
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
    parser.add_argument("--refinement-steps", type=int, default=1, help="number of iterations IC-STN shall perform to refine bbox proposals")
    parser.add_argument("--num-processes", type=int, help="number of processes to use for data loading")
    parser.add_argument("--use-serial-iterator", action='store_true', default=False, help="indicate that you do not want to use the multi process iterator")
    parser.add_argument("--refinement", action='store_true', default=False, help='enable param refinement with IC-STN')
    parser.add_argument("--render-all-bboxes", action='store_true', default=False, help="bbox plotter also renders all intermediate bboxes")
    parser = add_default_arguments(parser)
    args = parser.parse_args()

    image_size = Size(width=200, height=64)
    target_shape = Size(width=50, height=50)

    # attributes that need to be adjusted, once the Curriculum decides to use
    # a more difficult dataset
    # this is a 'map' of attribute name to path in trainer object
    attributes_to_adjust = [
        ('num_timesteps', ['predictor', 'localization_net']),
        ('num_timesteps', ['predictor', 'recognition_net']),
        ('num_timesteps', ['lossfun', '__self__']),
        ('num_labels', ['predictor', 'recognition_net']),
    ]

    with open(args.char_map, 'r') as fp:
        char_map = json.load(fp)
    num_labels = len(char_map)
    curriculum = BabyStepCurriculum(
        args.dataset_specification,
        TextRecFileDataset,
        args.blank_label,
        args.gpus,
        attributes_to_adjust=attributes_to_adjust,
        trigger=(args.test_interval, 'iteration'),
        min_delta=1.0,
        dataset_args={
            'char_map': args.char_map,
            'resize_size': target_shape,
            'blank_label': args.blank_label,
        }
    )

    train_dataset, validation_dataset = curriculum.load_dataset(0)
    train_dataset.resize_size = image_size
    validation_dataset.resize_size = image_size

    metrics = TextRecSoftmaxMetrics(
        args.blank_label,
        args.char_map,
        train_dataset.num_timesteps,
        image_size,
        area_loss_factor=args.area_factor,
        aspect_ratio_loss_factor=args.aspect_factor,
        area_scaling_factor=args.area_scale_factor,
    )

    localization_net = InverseCompositionalLocalizationNet(
        args.dropout_ratio,
        train_dataset.num_timesteps,
        args.refinement_steps,
        target_shape,
        zoom=args.zoom,
        do_parameter_refinement=args.refinement
    )
    recognition_net = TextRecognitionNet(
        target_shape,
        num_rois=train_dataset.num_timesteps,
        label_size=num_labels,
    )
    net = TextRecNet(localization_net, recognition_net)

    model = Classifier(net, ('accuracy',), lossfun=metrics.calc_loss, accfun=metrics.calc_accuracy,
                       provide_label_during_forward=False)

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

    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    optimizer.add_hook(chainer.optimizer.GradientClipping(2))

    # freeze localization net
    if args.freeze_localization:
        localization_net.disable_update()

    if len(args.gpus) > 1:
        gpu_datasets = split_dataset_n_random(train_dataset, len(args.gpus))
        if not len(gpu_datasets[0]) == len(gpu_datasets[-1]):
            adapted_second_split = split_dataset(gpu_datasets[-1], len(gpu_datasets[0]))[0]
            gpu_datasets[-1] = adapted_second_split
    else:
        gpu_datasets = [train_dataset]

    if args.use_serial_iterator:
        train_iterators = [chainer.iterators.SerialIterator(dataset, args.batch_size) for dataset in gpu_datasets]
        validation_iterator = chainer.iterators.SerialIterator(validation_dataset, args.batch_size)
    else:
        train_iterators = [
            MultiprocessIterator(dataset, args.batch_size, n_processes=args.num_processes)
            for dataset in gpu_datasets
        ]

        validation_iterator = MultiprocessIterator(
            validation_dataset,
            args.batch_size,
            n_processes=args.num_processes,
            repeat=False
        )

    updater = MultiprocessParallelUpdater(
        train_iterators,
        optimizer,
        devices=args.gpus,
        converter=get_concat_and_pad_examples(args.blank_label)
    )
    updater.setup_workers()

    log_dir = os.path.join(args.log_dir, "{}_{}".format(datetime.datetime.now().isoformat(), args.log_name))
    args.log_dir = log_dir

    # backup current file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(__file__, log_dir)

    # log all necessary configuration params
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

    # callback that logs report
    def log_postprocess(stats_cpu):
        if stats_cpu['iteration'] == args.log_interval:
            stats_cpu.update(report)


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

    FastEvaluator = get_fast_evaluator((args.test_interval, 'iteration'))
    evaluator = (
        FastEvaluator(
            validation_iterator,
            model,
            device=updater._devices[0],
            eval_func=lambda *args: model(*args),
            num_iterations=args.test_iterations,
            converter=get_concat_and_pad_examples(args.blank_label)
        ),
        (args.test_interval, 'iteration')
    )
    epoch_validation_iterator = copy.copy(validation_iterator)
    epoch_validation_iterator._repeat = False
    epoch_evaluator = (
        chainer.training.extensions.Evaluator(
            epoch_validation_iterator,
            model,
            device=updater._devices[0],
            converter=get_concat_and_pad_examples(args.blank_label),
        ),
        (1, 'epoch')
    )

    model_snapshotter = (
        extensions.snapshot_object(net, 'model_{.updater.iteration}.npz'), (args.snapshot_interval, 'iteration'))

    # bbox plotter test
    if not args.test_image:
        test_image = validation_dataset.get_example(0)[0]
    else:
        test_image = train_dataset.load_image(args.test_image)

    bbox_plotter = (TextRecBBOXPlotter(
        test_image,
        os.path.join(log_dir, 'boxes'),
        target_shape,
        metrics,
        send_bboxes=args.send_bboxes,
        upstream_port=args.port,
        visualization_anchors=[["localization_net", "vis_anchor"], ["recognition_net", "vis_anchor"]],
        render_extracted_rois=False,
        invoke_before_training=True,
        render_intermediate_bboxes=args.render_all_bboxes,
    ), (10, 'iteration'))

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
        ],
    )

    open_interactive_prompt(
        bbox_plotter=bbox_plotter[0],
        curriculum=curriculum,
    )

    trainer.run()
