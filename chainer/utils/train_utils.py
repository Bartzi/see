import chainer
import copy
import os

import six
from chainer import reporter as reporter_module
from chainer import variable
from chainer.dataset import convert, concat_examples
from chainer.training import extension
from chainer.training import extensions
import chainer.training.trigger as trigger_module
from chainer.training.extensions import Evaluator

from .logger import Logger


class AttributeUpdater(extension.Extension):

    def __init__(self, shift, attr='lr', trigger=(1, 'epoch')):
        self.shift = shift
        self.attr = attr
        self.trigger = trigger_module.get_trigger(trigger)

    def __call__(self, trainer):
        if self.trigger(trainer):
            optimizer = trainer.updater.get_optimizer('main')
            current_value = getattr(optimizer, self.attr)
            shifted_value = current_value * self.shift
            setattr(optimizer, self.attr, shifted_value)


class TwoStateLearningRateShifter(extension.Extension):

    CONTINUOS_SHIFT_STATE = 0
    INTERVAL_BASED_SHIFT_STATE = 1

    def __init__(self, start_lr, states):
        self.start_lr = start_lr
        self.lr = start_lr
        self.states = states
        self.current_state = self.states.pop(0)
        self.start_epoch = 0
        self.start_iteration = 0
        self.set_triggers()

    def set_triggers(self):
        self.target_lr = self.current_state['target_lr']
        self.update_trigger = trigger_module.get_trigger(self.current_state['update_trigger'])
        self.stop_trigger = trigger_module.get_trigger(self.current_state['stop_trigger'])
        self.phase_length, self.unit = self.current_state['stop_trigger']

    def switch_state_if_necessary(self, trainer):
        if self.stop_trigger(trainer):
            if len(self.states) > 1:
                self.current_state = self.states.pop(0)
                self.set_triggers()
                self.start_lr = self.target_lr
                self.start_epoch = trainer.updater.epoch
                self.start_iteration = self.update_trigger.iteration

    def __call__(self, trainer):
        updater = trainer.updater
        optimizer = trainer.updater.get_optimizer('main')

        if self.update_trigger(trainer):
            if self.current_state['state'] == self.CONTINUOS_SHIFT_STATE:
                epoch = updater.epoch_detail

                if self.unit == 'iteration':
                    interpolation_factor = (updater.iteration - self.start_iteration) / self.phase_length
                else:
                    interpolation_factor = (epoch - self.start_epoch) / self.phase_length

                new_lr = (1 - interpolation_factor) * self.start_lr + interpolation_factor * self.target_lr
                self.lr = new_lr
                optimizer.lr = new_lr

            else:
                optimizer.lr = self.target_lr
                self.lr = optimizer.lr

        self.switch_state_if_necessary(trainer)


class FastEvaluatorBase(Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, num_iterations=200):
        super(FastEvaluatorBase, self).__init__(
            iterator,
            target,
            converter=converter,
            device=device,
            eval_hook=eval_hook,
            eval_func=eval_func
        )
        self.num_iterations = num_iterations

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for _ in range(min(len(iterator.dataset) // iterator.batch_size, self.num_iterations)):
            batch = next(it, None)
            if batch is None:
                break

            observation = {}
            with reporter_module.report_scope(observation), chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    eval_func(*in_arrays)
                elif isinstance(in_arrays, dict):
                    eval_func(**in_arrays)
                else:
                    eval_func(in_arrays)

            summary.add(observation)

        return summary.compute_mean()


def get_fast_evaluator(trigger_interval):
    return type('FastEvaluator', (FastEvaluatorBase,), dict(trigger=trigger_interval, name='fast_validation'))


def get_trainer(net, updater, log_dir, print_fields, extra_extensions=(), epochs=10, snapshot_interval=20000, print_interval=100, postprocess=None, do_logging=True, model_files=()):
    trainer = chainer.training.Trainer(
        updater,
        (epochs, 'epoch'),
        out=log_dir,
    )

    # dump computational graph
    trainer.extend(extensions.dump_graph('main/loss'))

    # also observe learning rate
    observe_lr_extension = chainer.training.extensions.observe_lr()
    observe_lr_extension.trigger = (print_interval, 'iteration')
    trainer.extend(observe_lr_extension)

    # Take snapshots
    trainer.extend(
        extensions.snapshot(filename="trainer_snapshot"),
        trigger=lambda trainer:
        trainer.updater.is_new_epoch or
        (trainer.updater.iteration > 0 and trainer.updater.iteration % snapshot_interval == 0)
    )

    if do_logging:
        # write all statistics to a file
        trainer.extend(Logger(model_files, log_dir, keys=print_fields, trigger=(print_interval, 'iteration'), postprocess=postprocess))

        # print some interesting statistics
        trainer.extend(extensions.PrintReport(
            print_fields,
            log_report='Logger',
        ))

    # Progressbar!!
    trainer.extend(extensions.ProgressBar(update_interval=1))

    for extra_extension, trigger in extra_extensions:
        trainer.extend(extra_extension, trigger=trigger)

    return trainer


def add_default_arguments(parser):
    parser.add_argument("log_dir", help='directory where generated models and logs shall be stored')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, required=True,
                        help="Number of images per training batch")
    parser.add_argument('-g', '--gpus', type=int, nargs="*", default=[], help="Ids of GPU to use [default: (use cpu)]")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to train [default: 10]")
    parser.add_argument('-r', '--resume', help="path to previously saved state of trained model from which training shall resume")
    parser.add_argument('-si', '--snapshot-interval', dest='snapshot_interval', type=int, default=20000,
                        help="number of iterations after which a snapshot shall be taken [default: 20000]")
    parser.add_argument('-ln', '--log-name', dest='log_name', default='training', help="name of the log folder")
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=0.01,
                        help="initial learning rate [default: 0.01]")
    parser.add_argument('-li', '--log-interval', dest='log_interval', type=int, default=100,
                        help="number of iterations after which an update shall be logged [default: 100]")
    parser.add_argument('--lr-step', dest='learning_rate_step_size', type=float, default=0.1,
                        help="Step size for decreasing learning rate [default: 0.1]")
    parser.add_argument('-t', '--test-interval', dest='test_interval', type=int, default=1000,
                        help="number of iterations after which testing should be performed [default: 1000]")
    parser.add_argument('--test-iterations', dest='test_iterations', type=int, default=200,
                        help="number of test iterations [default: 200]")
    parser.add_argument("-dr", "--dropout-ratio", dest='dropout_ratio', default=0.5, type=float,
                        help="ratio for dropout layers")

    return parser


def get_concat_and_pad_examples(padding=-10000):
    def concat_and_pad_examples(batch, device=None):
        return concat_examples(batch, device=device, padding=padding)

    return concat_and_pad_examples


def concat_and_pad_examples(batch, device=None, padding=-10000):
    return concat_examples(batch, device=device, padding=padding)


def get_definition_filepath(obj):
    return __import__(obj.__module__, fromlist=obj.__module__.split('.')[:1]).__file__


def get_definition_filename(obj):
    return os.path.basename(get_definition_filepath(obj))

