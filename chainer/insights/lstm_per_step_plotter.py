import chainer.functions as F

from insights.bbox_plotter import BBOXPlotter


class LSTMPerStepBBOXPlotter(BBOXPlotter):

    def decode_predictions(self, predictions, xp):
        predictions = F.concat(predictions, axis=0)
        predictions = xp.argmax(predictions.data, axis=1)
        word = self.loss_metrics.strip_prediction(predictions[xp.newaxis, ...])[0]
        if len(word) == 1 and word[0] == 0:
            return ""
        word = "".join(map(self.loss_metrics.label_to_char, word))
        word = word.replace(chr(self.loss_metrics.char_map[str(self.loss_metrics.blank_symbol)]), '')
        return word
