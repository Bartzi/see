import chainer.functions as F

from metrics.loss_metrics import LossMetrics


class SoftmaxMetrics(LossMetrics):

    def calc_actual_loss(self, predictions, grid, labels):
        losses = []
        for char_prediction, char_gt in zip(F.separate(predictions, axis=0), F.separate(labels, axis=1)):
            losses.append(F.softmax_cross_entropy(char_prediction, char_gt))
        return sum(losses)
