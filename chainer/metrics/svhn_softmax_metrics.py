from chainer import cuda
import chainer.functions as F
from metrics.softmax_metrics import SoftmaxMetrics


class SVHNSoftmaxMetrics(SoftmaxMetrics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_loss(self, x, t):
        batch_predictions, _, _ = x

        # concat all individual predictions and slice for each time step
        batch_predictions = F.concat([F.expand_dims(p, axis=2) for p in batch_predictions], axis=2)

        self.xp = cuda.get_array_module(batch_predictions[0], t)
        batch_size = t.shape[0]
        t = F.reshape(t, (batch_size, self.num_timesteps, -1))

        losses = []
        for predictions, labels in zip(F.separate(batch_predictions, axis=0), F.separate(t, axis=1)):
            batch_size, num_chars, num_classes = predictions.shape
            predictions = F.reshape(predictions, (batch_size * num_chars, num_classes))
            labels = F.reshape(labels, (-1,))
            losses.append(F.softmax_cross_entropy(predictions, labels))

        return sum(losses)

    def calc_accuracy(self, x, t):
        batch_predictions, _, _ = x

        # concat all individual predictions and slice for each time step
        batch_predictions = F.concat([F.expand_dims(p, axis=2) for p in batch_predictions], axis=2)

        self.xp = cuda.get_array_module(batch_predictions[0], t)
        batch_size = t.shape[0]
        t = F.reshape(t, (batch_size, self.num_timesteps, -1))

        accuracies = []
        with cuda.get_device_from_array(batch_predictions.data):
            for prediction, label in zip(F.separate(batch_predictions, axis=0), F.separate(t, axis=1)):
                classification = F.softmax(prediction, axis=2)
                classification = classification.data
                classification = self.xp.argmax(classification, axis=2)
                # classification = self.xp.transpose(classification, (1, 0))

                words = self.strip_prediction(classification)
                labels = self.strip_prediction(label.data)

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
