import itertools

import chainer.functions as F
import chainer.links as L

from chainer import cuda

from metrics.loss_metrics import LossMetrics


class PerStepLSTMMetric(LossMetrics):

    def calc_actual_loss(self, predictions, grid, labels):
        pass

    def calc_loss(self, x, t):
        batch_predictions, _, grids = x
        self.xp = cuda.get_array_module(batch_predictions[0], t)

        # reshape labels
        batch_size = t.shape[0]

        # reshape grids
        grid_shape = grids.shape
        if self.uses_original_data:
            grids = F.reshape(grids, (self.num_timesteps, batch_size, 4,) + grid_shape[1:])
        else:
            grids = F.reshape(grids, (self.num_timesteps, batch_size, 1,) + grid_shape[1:])
        recognition_losses = []

        for prediction, label in zip(batch_predictions, F.separate(t, axis=1)):
            recognition_loss = F.softmax_cross_entropy(prediction, label)
            recognition_losses.append(recognition_loss)

        losses = [sum(recognition_losses) / len(recognition_losses)]

        # with cuda.get_device_from_array(grids.data):
        #     grid_list = F.separate(F.reshape(grids, (self.timesteps, -1,) + grids.shape[3:]), axis=0)
        #     overlap_losses = []
        #     for grid_1, grid_2 in itertools.combinations(grid_list, 2):
        #         overlap_losses.append(self.calc_iou_loss(grid_1, grid_2))
        #     losses.append(sum(overlap_losses) / len(overlap_losses))

        for i, grid in enumerate(F.separate(grids, axis=0), start=1):
            with cuda.get_device_from_array(grid.data):
                grid_losses = []
                for sub_grid in F.separate(grid, axis=1):
                    width, height = self.get_bbox_side_lengths(sub_grid)
                    grid_losses.append(self.area_loss_factor * self.calc_area_loss(width, height))
                    grid_losses.append(self.aspect_ratio_loss_factor * self.calc_aspect_ratio_loss(width, height))
                    grid_losses.append(self.calc_direction_loss(sub_grid))
                    grid_losses.append(self.calc_height_loss(height))
                losses.append(sum(grid_losses))

        return sum(losses) / len(losses)

    def calc_accuracy(self, x, t):
        batch_predictions, _, _ = x
        self.xp = cuda.get_array_module(batch_predictions[0], t)
        accuracies = []
        for prediction, label in zip(batch_predictions, F.separate(t, axis=1)):
            recognition_accuracy = F.accuracy(prediction, label)
            accuracies.append(recognition_accuracy)
        return sum(accuracies) / len(accuracies)
