from chainer import Chain, cuda
import chainer.functions as F
import chainer.links as L

from functions.rotation_droput import rotation_dropout
from insights.visual_backprop import VisualBackprop
from models.fsns import ResnetBlock


class SVHNLocalizationNet(Chain):

    def __init__(self, dropout_ratio, num_timesteps, zoom=0.9):
        super(SVHNLocalizationNet, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32)
            self.rs2 = ResnetBlock(48, filter_increase=True)
            self.rs3 = ResnetBlock(48)
            self.lstm = L.LSTM(None, 256)
            self.transform_2 = L.Linear(256, 6)

        # initialize transform
        self.transform_2.W.data[...] = 0

        transform_bias = self.transform_2.b.data
        transform_bias[[0, 4]] = zoom
        transform_bias[[2, 5]] = 0

        self.dropout_ratio = dropout_ratio
        self._train = True
        self.num_timesteps = num_timesteps
        self.vis_anchor = None

        self.width_encoding = None
        self.height_encoding = None

    def __call__(self, images):
        self.lstm.reset_state()

        h = self.bn0(self.conv0(images))
        h = F.average_pooling_2d(F.relu(h), 2, stride=2)

        h = self.rs1(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.rs3(h)
        # h = self.rs4(h)
        self.vis_anchor = h
        h = F.average_pooling_2d(h, 5, stride=2)

        localizations = []

        with cuda.get_device_from_array(h.data):
            for _ in range(self.num_timesteps):
                in_feature = h
                lstm_prediction = F.relu(self.lstm(in_feature))
                transformed = self.transform_2(lstm_prediction)
                transformed = F.reshape(transformed, (-1, 2, 3))
                localizations.append(rotation_dropout(transformed, ratio=self.dropout_ratio))

        return F.concat(localizations, axis=0)


class SVHNRecognitionNet(Chain):

    def __init__(self, target_shape, num_labels, num_timesteps, use_blstm=False):
        super(SVHNRecognitionNet, self).__init__()
        with self.init_scope():
            self.data_bn = L.BatchNormalization(3)
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
            self.classifier = L.Linear(None, 11)

        self._train = True
        self.target_shape = target_shape
        self.num_labels = num_labels
        self.num_timesteps = num_timesteps
        self.vis_anchor = None
        self.use_blstm = use_blstm

    def __call__(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        h = self.data_bn(rois)
        h = F.relu(self.bn0(self.conv0(h)))
        h = F.average_pooling_2d(h, 2, stride=2)

        h = self.rs1(h)
        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.rs3(h)
        self.vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        h = F.relu(self.fc1(h))

        # for each timestep of the localization net do the 'classification'
        h = F.reshape(h, (self.num_timesteps, -1, self.fc1.out_size))
        overall_predictions = []
        for timestep in F.separate(h, axis=0):
            lstm_predictions = []
            self.lstm.reset_state()
            if self.use_blstm:
                self.blstm.reset_state()

            for _ in range(self.num_labels):
                lstm_prediction = self.lstm(timestep)
                lstm_predictions.append(lstm_prediction)

            if self.use_blstm:
                blstm_predictions = []
                for lstm_prediction in reversed(lstm_predictions):
                    blstm_prediction = self.blstm(lstm_prediction)
                    blstm_predictions.append(blstm_prediction)

                lstm_predictions = reversed(blstm_predictions)

            final_lstm_predictions = []
            for lstm_prediction in lstm_predictions:
                classified = self.classifier(lstm_prediction)
                final_lstm_predictions.append(F.expand_dims(classified, axis=0))

            final_lstm_predictions = F.concat(final_lstm_predictions, axis=0)
            overall_predictions.append(final_lstm_predictions)

        return overall_predictions, rois, points


class SVHNCTCRecognitionNet(Chain):

    def __init__(self, target_shape, num_labels, num_timesteps):
        super(SVHNCTCRecognitionNet, self).__init__()
        with self.init_scope():
            self.data_bn = L.BatchNormalization(3)
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32)
            self.rs2 = ResnetBlock(64, filter_increase=True)
            self.rs3 = ResnetBlock(128, filter_increase=True)
            self.fc1 = L.Linear(None, 256)
            self.lstm = L.LSTM(None, 256)
            self.classifier = L.Linear(None, 11)

        self._train = True
        self.target_shape = target_shape
        self.num_labels = num_labels
        self.num_timesteps = num_timesteps
        self.vis_anchor = None

    def __call__(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        # h = self.data_bn(rois)
        h = F.relu(self.bn0(self.conv0(rois)))
        h = F.average_pooling_2d(h, 2, stride=2)

        h = self.rs1(h)
        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.rs3(h)
        self.vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        h = F.relu(self.fc1(h))

        # for each timestep of the localization net do the 'classification'
        h = F.reshape(h, (self.num_timesteps * 2 + 1, -1, self.fc1.out_size))
        overall_predictions = []
        for timestep in F.separate(h, axis=0):
            # go 2x num_labels plus 1 timesteps because of ctc loss
            lstm_predictions = []
            self.lstm.reset_state()
            for _ in range(self.num_labels):
                lstm_prediction = self.lstm(timestep)
                classified = self.classifier(lstm_prediction)
                lstm_predictions.append(classified)
            overall_predictions.append(lstm_predictions)

        return overall_predictions, rois, points


class SVHNNet(Chain):

    def __init__(self, localization_net, recognition_net):
        super(SVHNNet, self).__init__()
        with self.init_scope():
            self.localization_net = localization_net
            self.recognition_net = recognition_net

    def __call__(self, images):
        batch_size = images.shape[0]
        h = self.localization_net(images)
        new_batch_size = h.shape[0]
        batch_size_increase_factor = new_batch_size // batch_size
        images = F.concat([images for _ in range(batch_size_increase_factor)], axis=0)

        return self.recognition_net(images, h)
