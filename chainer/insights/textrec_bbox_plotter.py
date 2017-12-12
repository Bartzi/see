from chainer import cuda

import chainer.functions as F

from insights.bbox_plotter import BBOXPlotter


class TextRectBBoxPlotter(BBOXPlotter):

    def decode_predictions(self, predictions):
        # concat all individual predictions and slice for each time step
        predictions = predictions[0]

        with cuda.get_device_from_array(predictions.data):
            prediction = F.squeeze(predictions, axis=1)
            classification = F.softmax(prediction, axis=1)
            classification = classification.data
            classification = self.xp.argmax(classification, axis=1)

            words = self.loss_metrics.strip_prediction(classification[self.xp.newaxis, ...])[0]
            word = "".join(map(self.loss_metrics.label_to_char, words))

        return word
