from chainer import Function
from chainer.utils import type_check, force_array


class DisableTranslation(Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        grid_type,  = in_types

        type_check.expect(
            grid_type.dtype.kind == 'f',
            grid_type.ndim == 3,
            grid_type.shape[1] == 2,
            grid_type.shape[2] == 3,
        )

    def forward(self, inputs):
        self.retain_inputs(())
        grid, = inputs
        grid[:, :, 2] = 0

        return grid,

    def backward(self, inputs, grad_outputs):
        grad_outputs = grad_outputs[0]
        grad_outputs[:, :, 2] = 0
        return grad_outputs,


def disable_translation(grid):
    return DisableTranslation()(grid)
