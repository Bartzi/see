import chainer.functions as F
import chainer.links as L

from chainer import Chain

from models.fsns import ResnetBlock


class TextRecognitionNet(Chain):

    def __init__(self, target_shape, num_rois, label_size, use_blstm=False):
        super().__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1, stride=2)
            self.bn0 = L.BatchNormalization(32)
            self.conv1 = L.Convolution2D(32, 32, 3, pad=1)
            self.bn1 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32)
            self.rs2 = ResnetBlock(64, filter_increase=True)
            self.rs3 = ResnetBlock(128, filter_increase=True)
            self.fc1 = L.Linear(None, 256)
            self.lstm = L.LSTM(None, 256)
            if use_blstm:
                 self.blstm = L.LSTM(None, 256)
            self.classifier = L.Linear(None, label_size)

        self.use_blstm = use_blstm
        self.target_shape = target_shape
        self.num_rois = num_rois

    def __call__(self, images, localizations):
        self.lstm.reset_state()
        if self.use_blstm:
            self.blstm.reset_state()

        points = [F.spatial_transformer_grid(localization, self.target_shape) for localization in localizations]
        rois = [F.spatial_transformer_sampler(images, point) for point in points]

        h = F.relu(self.bn0(self.conv0(rois[-1])))
        h = F.average_pooling_2d(h, 2, stride=2)

        h = self.rs1(h)
        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.rs3(h)
        self.vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        h = F.relu(self.fc1(h))

        # each timestep of the localization contains one character prediction, that needs to be classified
        overall_predictions = []
        h = F.reshape(h, (self.num_rois, -1, self.fc1.out_size))

        for timestep in F.separate(h, axis=0):
            lstm_state = self.lstm(timestep)

            prediction = self.classifier(lstm_state)
            overall_predictions.append(prediction)

        return overall_predictions, rois, points


class TextRecNet(Chain):

    def __init__(self, localization_net, recognition_net):
        super(TextRecNet, self).__init__()
        with self.init_scope():
            self.localization_net = localization_net
            self.recognition_net = recognition_net

    def __call__(self, images):
        batch_size = images.shape[0]
        h = self.localization_net(images)
        new_batch_size = h[-1].shape[0]
        batch_size_increase_factor = new_batch_size // batch_size
        images = F.concat([images for _ in range(batch_size_increase_factor)], axis=0)

        return self.recognition_net(images, h)
