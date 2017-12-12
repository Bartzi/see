import json
import statistics
from collections import deque

import copy
from chainer import cuda, Variable
from chainer.training import extension
import chainer.training.trigger as trigger_module

from datasets.concatenated_dataset import ConcatenatedDataset
from datasets.sub_dataset import split_dataset_random, split_dataset


class BabyStepCurriculum(extension.Extension):

    def __init__(self, dataset_specification, dataset_class, blank_label, trigger=(1, 'epoch'), min_delta=0.01, attributes_to_adjust=(), maxlen=5, dataset_args=None):
        if dataset_args is None:
            dataset_args = {}
        self.dataset_class = dataset_class
        self.dataset_args = dataset_args
        self.trigger = trigger_module.get_trigger(trigger)
        self.maxlen = maxlen
        self.queue = deque(maxlen=self.maxlen)
        self.min_delta = min_delta
        self.attributes_to_adjust = attributes_to_adjust
        self.blank_label = blank_label
        self.force_enlarge_dataset = False

        with open(dataset_specification) as specification:
            specification = json.load(specification)

            self.train_curriculum = {i: s['train'] for i, s in enumerate(specification)}
            self.validation_curriculum = {i: s['validation'] for i, s in enumerate(specification)}

        self.current_level = 0

    def load_dataset(self, level):
        train_dataset = self.dataset_class(self.train_curriculum[level], **self.dataset_args)
        validation_dataset = self.dataset_class(self.validation_curriculum[level], **self.dataset_args)

        return train_dataset, validation_dataset

    def training_converged(self):
        # check whether system already settled and we can enlarge train set
        reference_value = self.queue.pop()
        deltas = []
        for value in self.queue:
            deltas.append(abs(value - reference_value))

        mean = statistics.mean(deltas)
        return mean <= self.min_delta

    def adjust_attributes(self, model, dataset):
        for attribute_name, attribute_path in self.attributes_to_adjust:
            chain = model
            # find the correct chain/link in our model as provided by attribute path
            for path in attribute_path:
                chain = getattr(chain, path)
            # set the corresponding attribute of our chain/link, with the attribute provided by the given dataset
            setattr(chain, attribute_name, getattr(dataset, attribute_name))

    def __call__(self, trainer):
        if self.force_enlarge_dataset:
            self.force_enlarge_dataset = False
            self.enlarge_dataset(trainer)

        if self.trigger(trainer):
            with cuda.get_device_from_id(trainer.updater.get_optimizer('main').target._device_id):
                loss = trainer.observation.get('validation/main/loss', None)
                if loss is None:
                    return
                queue_data = loss.data if isinstance(loss, Variable) else loss
                self.queue.append(float(queue_data))
                if len(self.queue) >= self.maxlen:
                    if not self.training_converged():
                        return

                    self.enlarge_dataset(trainer)

    def enlarge_dataset(self, trainer):
        print("enlarging datasets")
        # we can add new samples to the train dataset
        self.current_level += 1
        try:
            train_dataset, validation_dataset = self.load_dataset(self.current_level)
        except KeyError:
            # we have exhausted our train curriculum we need to stop!
            raise StopIteration
        self.update_iterators(trainer, train_dataset, validation_dataset)
        self.adjust_attributes(trainer.updater.get_optimizer('main').target, train_dataset)
        self.queue.clear()

    @staticmethod
    def split_dataset(dataset):
        gpu_datasets = split_dataset_random(dataset, len(dataset) // 2)
        if not len(gpu_datasets[0]) == len(gpu_datasets[1]):
            adapted_second_split = split_dataset(gpu_datasets[1], len(gpu_datasets[0]))[0]
            gpu_datasets = (gpu_datasets[0], adapted_second_split)
        return gpu_datasets

    def pad_dataset(self, old_dataset, new_dataset):
        old_dataset.pad_labels(new_dataset.num_timesteps, self.blank_label)
        return old_dataset

    def update_iterators(self, trainer, train_dataset, validation_dataset):
        train_iterators = getattr(trainer.updater, '_mpu_iterators', None)
        if train_iterators is None:
            train_iterators = [trainer.updater.get_iterator('main')]

        validation_iterator = trainer.get_extension('fast_validation').get_iterator('main')

        # pad old dataset
        for train_iterator in train_iterators:
            train_iterator.dataset = self.pad_dataset(train_iterator.dataset, train_dataset)
        validation_iterator.dataset = self.pad_dataset(validation_iterator.dataset, validation_dataset)

        # concatenate new dataset with old dataset
        new_train_datasets = []
        if len(train_iterators) > 1:
            for iterator, dataset in zip(train_iterators, self.split_dataset(train_dataset)):
                new_train_datasets.append(ConcatenatedDataset(dataset, iterator.dataset))
        else:
            new_train_datasets.append(ConcatenatedDataset(train_dataset, train_iterators[0].dataset))
        new_validation_dataset = ConcatenatedDataset(validation_dataset, validation_iterator.dataset)

        # create new iterator
        new_train_iterators = [iterator.__class__(
            dataset,
            iterator.batch_size,
        ) for iterator, dataset in zip(train_iterators, new_train_datasets)]

        new_validation_iterator = validation_iterator.__class__(
            new_validation_dataset,
            validation_iterator.batch_size,
        )

        # exchange iterators in trainer
        if hasattr(trainer.updater, '_mpu_iterators'):
            # for iterator, worker in zip(new_train_iterators, trainer.updater._workers):
            #     worker.iterator = iterator
            trainer.updater._mpu_iterators = new_train_iterators
        trainer.updater._iterators['main'] = new_train_iterators[0]
        trainer.get_extension('fast_validation')._iterators['main'] = new_validation_iterator

        # in case we have a real validation extension, not just our fast evaluator we also need to change the iterator there
        try:
            validator = trainer.get_extension('validation')
            copy_of_new_validation_iterator = copy.copy(new_validation_iterator)
            copy_of_new_validation_iterator._repeat = False
            validator._iterators['main'] = copy_of_new_validation_iterator
        except KeyError:
            pass

        for iterator in [*train_iterators, validation_iterator]:
            iterator.finalize()

