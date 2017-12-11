from chainer import reporter
from chainer.functions import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer.links import Classifier as OriginalClassifier


class Classifier(OriginalClassifier):
    """
        Classifier that is able to log two different accuracies
    """

    def __init__(self, predictor, accuracy_types,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy, accfun=accuracy, provide_label_during_forward=False):
        super(Classifier, self).__init__(predictor, lossfun=lossfun, accfun=accfun)
        assert type(accuracy_types) is tuple, "accuracy_types must be a tuple of strings"
        self.accuracy_types = accuracy_types
        self.provide_label_during_forward = provide_label_during_forward

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        if self.provide_label_during_forward:
            self.y = self.predictor(*x, t)
        else:
            self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            reported_accuracies = self.accfun(self.y, t)
            if len(self.accuracy_types) == 1:
                reported_accuracies = reported_accuracies,
            report = {accuracy_type: reported_accuracy
                      for accuracy_type, reported_accuracy in zip(self.accuracy_types, reported_accuracies)}
            reporter.report(report, self)
        return self.loss
