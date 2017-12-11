from chainer import cuda

import chainer.functions as F

from metrics.loss_metrics import LossMetrics


class CTCMetrics(LossMetrics):

    def calc_actual_loss(self, predictions, grid, labels):
        loss = F.connectionist_temporal_classification(predictions, labels, self.blank_symbol)
        return loss

    def strip_prediction(self, predictions):
        # TODO Parallelize
        words = []
        for prediction in predictions:
            blank_symbol_seen = False
            stripped_prediction = self.xp.full((1,), prediction[0], dtype=self.xp.int32)
            for char in prediction:
                if char == self.blank_symbol:
                    blank_symbol_seen = True
                    continue
                if char == stripped_prediction[-1] and not blank_symbol_seen:
                    continue
                blank_symbol_seen = False
                stripped_prediction = self.xp.hstack((stripped_prediction, char.reshape(1,)))
            words.append(stripped_prediction)
        return words
