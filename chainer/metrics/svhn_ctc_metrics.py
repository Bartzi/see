from chainer import cuda
import chainer.functions as F
from metrics.loss_metrics import LossMetrics


class SVHNCTCMetrics(LossMetrics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_num_timesteps = 2 * self.num_timesteps + 1

    def calc_actual_loss(self, predictions, grid, labels):
        predictions = F.separate(predictions, axis=0)
        return F.connectionist_temporal_classification(predictions, labels, blank_symbol=self.blank_symbol)

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
                stripped_prediction = self.xp.hstack((stripped_prediction, char.reshape(1, )))
            words.append(stripped_prediction)
        return words


