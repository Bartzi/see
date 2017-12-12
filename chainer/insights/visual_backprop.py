import chainer
from chainer import cuda
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.pooling.pooling_2d import Pooling2D

import chainer.functions as F
from chainer.training import Extension


class VisualBackprop(Extension):

    def __init__(self):
        super().__init__()
        self.xp = None

    def traverse_computational_graph(self, node, feature_map):
        if node.inputs[0].creator is None:
            return feature_map

        if isinstance(node, Convolution2DFunction) or isinstance(node, Pooling2D):
            feature_map = self.scale_layer(feature_map, node)

        return self.traverse_computational_graph(node.inputs[0].creator, feature_map)

    def scale_layer(self, feature_map, node):
        input_data = node.inputs[0].data
        _, _, in_height, in_width = input_data.shape
        _, _, feature_height, feature_width = feature_map.shape
        kernel_height = in_height + 2 * node.ph - node.sy * (feature_height - 1)
        kernel_width = in_width + 2 * node.pw - node.sx * (feature_width - 1)
        scaled_feature = F.deconvolution_2d(
            feature_map,
            self.xp.ones((1, 1, kernel_height, kernel_width)),
            stride=(node.sy, node.sx),
            pad=(node.ph, node.pw),
            outsize=(in_height, in_width),
        )
        averaged_feature_map = F.average(input_data, axis=1, keepdims=True)
        feature_map = scaled_feature * averaged_feature_map
        return feature_map

    def perform_visual_backprop(self, variable):
        with chainer.no_backprop_mode(), chainer.cuda.get_device_from_array(variable.data):
            self.xp = cuda.get_array_module(variable)
            averaged_feature = F.average(variable, axis=1, keepdims=True)

            visualization = self.traverse_computational_graph(variable.creator, averaged_feature)
            visualization = visualization.data
            for i in range(len(visualization)):
                min_val = visualization[i].min()
                max_val = visualization[i].max()
                visualization[i] -= min_val
                visualization[i] *= 1.0 / (max_val - min_val)
        return visualization
