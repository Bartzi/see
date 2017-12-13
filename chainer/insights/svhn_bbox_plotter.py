from chainer import cuda

import chainer.functions as F

from insights.bbox_plotter import BBOXPlotter


class SVHNBBoxPlotter(BBOXPlotter):

    def decode_predictions(self, predictions):
        # concat all individual predictions and slice for each time step
        predictions = F.split_axis(predictions, 4, 0)
        predictions = F.concat([F.expand_dims(p, axis=2) for p in predictions], axis=2)

        words = []
        with cuda.get_device_from_array(predictions.data):
            for prediction in F.separate(predictions, axis=0):
                prediction = F.squeeze(prediction, axis=0)
                prediction = F.softmax(prediction, axis=1)
                prediction = self.xp.argmax(prediction.data, axis=1)
                word = self.loss_metrics.strip_prediction(prediction[self.xp.newaxis, ...])[0]
                if len(word) == 1 and word[0] == 0:
                    return ''

                word = "".join(map(self.loss_metrics.label_to_char, word))
                word = word.replace(chr(self.loss_metrics.char_map[str(self.loss_metrics.blank_symbol)]), '')
                words.append(word)

        text = " ".join(words)
        return text
