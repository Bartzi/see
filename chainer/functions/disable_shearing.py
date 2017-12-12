from chainer import Function
from chainer.utils import type_check, force_array


class DisableShearing(Function):

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
        grid, = inputs

        grid[:, 0, 1] = 0
        grid[:, 1, 0] = 0

        return force_array(grid, dtype=grid.dtype),

    def backward(self, inputs, grad_outputs):
        grad_outputs = grad_outputs[0]
        grad_outputs[:, 0, 1] = 0
        grad_outputs[:, 1, 0] = 0
        return force_array(grad_outputs, dtype=inputs[0].dtype),


def disable_shearing(grid):
    return DisableShearing()(grid)
