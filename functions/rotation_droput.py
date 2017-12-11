import numpy

from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class RotationDropout(function.Function):

    """Dropout regularsation for training rotation of spatial transformer"""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 3,
            x_type.shape[1] == 2,
            x_type.shape[2] == 3,
        )

    def forward(self, x):
        self.retain_inputs(())
        xp = cuda.get_array_module(*x)

        if not configuration.config.train:
            # scale affected weights if we are testing
            mask = xp.ones_like(x[0])
            mask[:, 0, 1] = self.dropout_ratio
            mask[:, 1, 0] = self.dropout_ratio

            return x[0] * mask,

        if not hasattr(self, 'mask'):
            self.mask = xp.ones_like(x[0])

            flag_data = xp.random.rand(1) < self.dropout_ratio
            self.mask[:, 0, 1] = flag_data
            self.mask[:, 1, 0] = flag_data

        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,


def rotation_dropout(x, ratio=.5, **kwargs):
    return RotationDropout(ratio)(x)
