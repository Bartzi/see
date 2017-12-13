import chainer.functions as F

from chainer import cuda

from metrics.loss_metrics import LossMetrics


class TextRectMetrics(LossMetrics):

    def calc_loss(self, x, t):
        batch_predictions, _, grids = x
        self.xp = cuda.get_array_module(batch_predictions, t)

        loss = self.calc_actual_loss(batch_predictions, None, t)

        # reshape grids
        batch_size = t.shape[0]
        grid_shape = grids.shape
        grids = F.reshape(grids, (-1, batch_size) + grid_shape[1:])

        grid_losses = []
        for grid in F.separate(grids, axis=0):
            with cuda.get_device_from_array(getattr(grid, 'data', grid[0].data)):
                grid_losses.append(self.calc_direction_loss(grid))

        return loss + (sum(grid_losses) / len(grid_losses))

    def calc_accuracy(self, x, t):
        batch_predictions, _, _ = x

        self.xp = cuda.get_array_module(batch_predictions[0], t)
        accuracies = []

        with cuda.get_device_from_array(batch_predictions.data):
            classification = F.softmax(batch_predictions, axis=2)
            classification = classification.data
            classification = self.xp.argmax(classification, axis=2)
            classification = self.xp.transpose(classification, (1, 0))

            words = self.strip_prediction(classification)
            labels = self.strip_prediction(t)

            num_correct_words = 0
            for word, label in zip(words, labels):
                word = "".join(map(self.label_to_char, word))
                label = "".join(map(self.label_to_char, label))
                if word == label:
                    num_correct_words += 1

            accuracy = num_correct_words / len(labels)
            accuracies.append(accuracy)

        overall_accuracy = sum(accuracies) / max(len(accuracies), 1)
        self.scale_area_loss_factor(overall_accuracy)
        return overall_accuracy


class TextRecCTCMetrics(TextRectMetrics):

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


class TextRecSoftmaxMetrics(TextRectMetrics):

    def calc_actual_loss(self, predictions, grid, labels):
        labels = F.reshape(labels, (-1,))

        predictions = F.transpose(predictions, (1, 0, 2))
        predictions = F.reshape(predictions, (-1, predictions.shape[-1]))
        return F.softmax_cross_entropy(predictions, labels)
