import statistics
from collections import deque

import math

from chainer import cuda, Variable
from chainer.training import extension
import chainer.training.trigger as trigger_module


class IntelligentAttributeShifter(extension.Extension):

    def __init__(self, shift, attr='lr', trigger=(1, 'iteration'), min_delta=0.1):
        self.shift = shift
        self.attr = attr
        self.trigger = trigger_module.get_trigger(trigger)
        self.queue = deque(maxlen=5)
        self.min_delta = min_delta
        self.force_shift = False

    def __call__(self, trainer):
        if self.force_shift:
            self.shift_attribute(trainer)
            self.force_shift = False
            return

        if self.trigger(trainer):
            with cuda.get_device_from_id(trainer.updater.get_optimizer('main').target._device_id):
                loss = trainer.observation.get('validation/main/loss', None)
                if loss is None:
                    return
                queue_data = loss.data if isinstance(loss, Variable) else loss
                self.queue.append(float(queue_data))
                if len(self.queue) == self.queue.maxlen:
                    # check whether we need to shift attribute
                    deltas = []
                    rotated_queue = self.queue.copy()
                    rotated_queue.rotate(-1)
                    rotated_queue.pop()
                    for element_1, element_2 in zip(self.queue, rotated_queue):
                        deltas.append(abs(element_1 - element_2))

                    delta = sum(deltas) / len(deltas)
                    # if change over last 5 validations was lower than min change shift attribute
                    if delta < self.min_delta:
                        self.shift_attribute(trainer)

    def shift_attribute(self, trainer):
        print("Shifting attribute {}".format(self.attr))
        optimizer = trainer.updater.get_optimizer('main')
        current_value = getattr(optimizer, self.attr)
        shifted_value = current_value * self.shift
        setattr(optimizer, self.attr, shifted_value)
        self.queue.clear()
