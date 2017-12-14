from chainer import Chain
from chainer import cuda

import chainer.functions as F
import chainer.links as L
from functions.disable_shearing import disable_shearing
from functions.disable_translation import disable_translation
from functions.rotation_droput import rotation_dropout
from insights.visual_backprop import VisualBackprop


class ResnetBlock(Chain):

    def __init__(self, num_filter, filter_increase=False, use_dropout=False, dropout_ratio=0.5):
        super().__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, num_filter, 3, pad=1)
            self.bn0 = L.BatchNormalization(num_filter)
            self.conv1 = L.Convolution2D(num_filter, num_filter, 3, pad=1)
            self.bn1 = L.BatchNormalization(num_filter)
            self.use_dropout = use_dropout
            self.dropout_ratio = dropout_ratio

            if filter_increase:
                self.conv2 = L.Convolution2D(None, num_filter, 1)
                self.bn2 = L.BatchNormalization(num_filter)

        self.filter_increase = filter_increase
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    def __call__(self, x):
        h = self.bn0(self.conv0(x))
        if self.use_dropout:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = F.relu(h)
        h = self.bn1(self.conv1(h))
        if self.use_dropout:
            h = F.dropout(h, ratio=self.dropout_ratio)

        if self.filter_increase:
            h_pre = self.bn2(self.conv2(x))
            h = h + h_pre

        h = F.relu(h)
        return h


class FSNSMultipleSTNLocalizationNet(Chain):

    def __init__(self, dropout_factor, num_timesteps, zoom=0.9):
        super(FSNSMultipleSTNLocalizationNet, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32)
            self.rs2 = ResnetBlock(48, filter_increase=True)
            self.rs3 = ResnetBlock(48)
            self.lstm = L.LSTM(None, 256)
            self.translation_transform = L.Linear(256, 6)
            self.rotation_transform = L.Linear(256, 6)
            self.transform_2 = L.LSTM(256, 6)

        self.dropout_factor = dropout_factor
        self._train = True
        self.num_timesteps = num_timesteps

        for transform in [self.translation_transform, self.rotation_transform]:
            transform_bias = transform.b.data
            transform_bias[[0, 4]] = zoom
            transform_bias[[2, 5]] = 0
            transform.W.data[...] = 0

        # self.transform_2.upward.b.data[...] = 0
        # self.transform_2.upward.W.data[...] = 0
        # self.transform_2.lateral.W.data[...] = 0

        # self.transform.W.data[...] = 0

        self.visual_backprop = VisualBackprop()
        self.vis_anchor = None

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.rs1.train = value
        self.rs2.train = value
        self.rs3.train = value

    def __call__(self, images):
        self.lstm.reset_state()
        self.transform_2.reset_state()

        h = self.bn0(self.conv0(images))
        h = F.average_pooling_2d(F.relu(h), 2, stride=2)

        h = self.rs1(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.rs3(h)
        self.vis_anchor = h
        h = F.average_pooling_2d(h, 5, stride=2)

        localizations = []

        with cuda.get_device_from_array(h.data):
            homogenuous_addon = self.xp.zeros((len(h), 1, 3), dtype=h.data.dtype)
            homogenuous_addon[:, 0, 2] = 1

        for _ in range(self.num_timesteps):
            lstm_prediction = F.relu(self.lstm(h))
            translation_transform = F.reshape(self.rotation_transform(lstm_prediction), (-1, 2, 3))
            translation_transform = disable_shearing(translation_transform)
            translation_transform = F.concat((translation_transform, homogenuous_addon), axis=1)

            rotation_transform = F.reshape(self.rotation_transform(lstm_prediction), (-1, 2, 3))
            rotation_transform = disable_translation(rotation_transform)
            rotation_transform = F.concat((rotation_transform, homogenuous_addon), axis=1)

            # first rotate, then translate
            transform = F.batch_matmul(rotation_transform, translation_transform)
            # homogenuous_multiplier = F.get_item(transform, (..., 2, 2))
            #
            # # bring matrices from homogenous coordinates to normal coordinates
            transform = transform[:, :2, :]
            # transform = transform / homogenuous_multiplier
            localizations.append(rotation_dropout(transform, ratio=self.dropout_factor))

        return F.concat(localizations, axis=0)


class FSNSSingleSTNLocalizationNet(Chain):

    def __init__(self, dropout_ratio, num_timesteps, zoom=0.9, use_dropout=False):
        super(FSNSSingleSTNLocalizationNet, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.rs2 = ResnetBlock(48, filter_increase=True, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.rs3 = ResnetBlock(48)
            # self.rs4 = ResnetBlock(16, filter_increase=True)
            self.lstm = L.LSTM(None, 256)
            self.transform_2 = L.LSTM(256, 6)

        self.dropout_ratio = dropout_ratio
        self.use_dropout = use_dropout
        self._train = True
        self.num_timesteps = num_timesteps

        # initialize transform
        # self.transform_2.W.data[...] = 0
        #
        # transform_bias = self.transform_2.b.data
        # transform_bias[[0, 4]] = zoom
        # transform_bias[[2, 5]] = 0

        self.visual_backprop = VisualBackprop()
        self.vis_anchor = None

        self.width_encoding = None
        self.height_encoding = None

    def __call__(self, images):
        self.lstm.reset_state()
        self.transform_2.reset_state()

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
            # lstm_prediction = chainer.Variable(self.xp.zeros((len(images), self.lstm.state_size), dtype=h.dtype))

            for _ in range(self.num_timesteps):
                # in_feature = self.attend(h, lstm_prediction)
                in_feature = h
                lstm_prediction = F.relu(self.lstm(in_feature))
                transformed = self.transform_2(lstm_prediction)
                transformed = F.reshape(transformed, (-1, 2, 3))
                localizations.append(rotation_dropout(transformed, ratio=self.dropout_ratio))

        return F.concat(localizations, axis=0)


class FSNSRecognitionNet(Chain):

    def __init__(self, target_shape, num_labels, num_timesteps, uses_original_data=False, dropout_ratio=0.5, use_dropout=False):
        super(FSNSRecognitionNet, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1, stride=2)
            self.bn0 = L.BatchNormalization(32)
            self.conv1 = L.Convolution2D(32, 32, 3, pad=1)
            self.bn1 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.rs2 = ResnetBlock(64, filter_increase=True, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.rs3 = ResnetBlock(128, filter_increase=True, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.fc1 = L.Linear(None, 256)
            self.lstm = L.LSTM(None, 256)
            self.classifier = L.Linear(None, 134)

        self._train = True
        self.target_shape = target_shape
        self.num_labels = num_labels
        self.num_timesteps = num_timesteps
        self.uses_original_data = uses_original_data
        self.vis_anchor = None
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

    def __call__(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        h = F.relu(self.bn0(self.conv0(rois)))
        if self.use_dropout:
            h = F.dropout(h, ratio=self.dropout_ratio)
        h = F.relu(self.bn1(self.conv1(h)))
        if self.use_dropout:
            h = F.dropout(h, ratio=self.dropout_ratio)

        h = self.rs1(h)
        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.rs3(h)
        self.vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        if self.uses_original_data:
            # merge data of all 4 individual images in channel dimension
            batch_size, num_channels, height, width = h.shape
            h = F.reshape(h, (batch_size // 4, 4 * num_channels, height, width))

        h = F.relu(self.fc1(h))

        # for each timestep of the localization net do the 'classification'
        h = F.reshape(h, (self.num_timesteps, -1, self.fc1.out_size))
        overall_predictions = []
        for timestep in F.separate(h, axis=0):
            # go 2x num_labels plus 1 timesteps because of ctc loss
            lstm_predictions = []
            self.lstm.reset_state()
            for _ in range(self.num_labels * 2 + 1):
                lstm_prediction = self.lstm(timestep)
                classified = self.classifier(lstm_prediction)
                lstm_predictions.append(classified)
            overall_predictions.append(lstm_predictions)

        return overall_predictions, rois, points


class FSNSSoftmaxRecognitionNet(Chain):

    def __init__(self, target_shape, num_labels, num_timesteps, uses_original_data=False, dropout_ratio=0.5, use_dropout=False, use_blstm=False):
        super().__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32)
            self.rs2 = ResnetBlock(64, filter_increase=True, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.rs3 = ResnetBlock(128, filter_increase=True, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
            self.fc1 = L.Linear(None, 256)
            self.lstm = L.LSTM(None, 256)
            if use_blstm:
                self.blstm = L.LSTM(None, 256)
            self.classifier = L.Linear(None, 134)

        self._train = True
        self.target_shape = target_shape
        self.num_labels = num_labels
        self.num_timesteps = num_timesteps
        self.uses_original_data = uses_original_data
        self.vis_anchor = None
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_blstm = use_blstm

    def __call__(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        h = self.bn0(self.conv0(rois))
        h = F.average_pooling_2d(F.relu(h), 2, stride=2)

        h = self.rs1(h)
        h = self.rs2(h)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.rs3(h)
        self.vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        if self.uses_original_data:
            # merge data of all 4 individual images in channel dimension
            batch_size, num_channels, height, width = h.shape
            h = F.reshape(h, (batch_size // 4, 4 * num_channels, height, width))

        h = F.relu(self.fc1(h))

        # for each timestep of the localization net do the 'classification'
        h = F.reshape(h, (self.num_timesteps, -1, self.fc1.out_size))
        overall_predictions = []
        for timestep in F.separate(h, axis=0):
            # go 2x num_labels plus 1 timesteps because of ctc loss
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


class FSNSSoftmaxRecognitionResNet(Chain):

    def __init__(self, target_shape, num_labels, num_timesteps, uses_original_data=False, dropout_ratio=0.5, use_dropout=False, use_blstm=False, use_attention=False):
        super().__init__()
        with self.init_scope():
            self.data_bn = L.BatchNormalization(3)
            self.conv0 = L.Convolution2D(None, 64, 7, stride=2, pad=3, nobias=True)
            self.bn0 = L.BatchNormalization(64)
            self.rs1_1 = ResnetBlock(64)
            self.rs1_2 = ResnetBlock(64)
            self.rs2_1 = ResnetBlock(128, filter_increase=True)
            self.rs2_2 = ResnetBlock(128)
            self.rs3_1 = ResnetBlock(256, filter_increase=True)
            self.rs3_2 = ResnetBlock(256)
            self.rs4_1 = ResnetBlock(512, filter_increase=True)
            self.rs4_2 = ResnetBlock(512)
            self.fc1 = L.Linear(None, 512)
            self.lstm = L.LSTM(None, 512)
            if use_blstm:
                self.blstm = L.LSTM(None, 512)
            if use_attention:
                self.transform_encoded_features = L.Linear(512, 512, nobias=True)
                self.transform_out_lstm_feature = L.Linear(512, 512, nobias=True)
                self.generate_attended_feat = L.Linear(512, 1)
                self.out_lstm = L.LSTM(512, 512)
            self.classifier = L.Linear(None, 134)

        self._train = True
        self.target_shape = target_shape
        self.num_labels = num_labels
        self.num_timesteps = num_timesteps
        self.uses_original_data = uses_original_data
        self.vis_anchor = None
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_blstm = use_blstm
        self.use_attention = use_attention

    def attend(self, encoded_features):
        self.out_lstm.reset_state()
        transformed_encoded_features = F.concat([F.expand_dims(self.transform_encoded_features(feature), axis=1) for feature in encoded_features], axis=1)
        concat_encoded_features = F.concat([F.expand_dims(e, axis=1) for e in encoded_features], axis=1)

        lstm_output = self.xp.zeros_like(encoded_features[0])
        outputs = []
        for _ in range(self.num_labels):
            transformed_lstm_output = self.transform_out_lstm_feature(lstm_output)
            attended_feats = []
            for transformed_encoded_feature in F.separate(transformed_encoded_features, axis=1):
                attended_feat = transformed_encoded_feature + transformed_lstm_output
                attended_feat = F.tanh(attended_feat)
                attended_feats.append(self.generate_attended_feat(attended_feat))

            attended_feats = F.concat(attended_feats, axis=1)
            alphas = F.softmax(attended_feats, axis=1)

            lstm_input_feature = F.batch_matmul(alphas, concat_encoded_features, transa=True)
            lstm_input_feature = F.squeeze(lstm_input_feature, axis=1)
            lstm_output = self.out_lstm(lstm_input_feature)
            outputs.append(lstm_output)
        return outputs

    def __call__(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        connected_rois = self.data_bn(rois)
        h = F.relu(self.bn0(self.conv0(connected_rois)))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)

        h = self.rs1_1(h)
        h = self.rs1_2(h)

        h = self.rs2_1(h)
        h = self.rs2_2(h)

        h = self.rs3_1(h)
        h = self.rs3_2(h)

        h = self.rs4_1(h)
        h = self.rs4_2(h)

        self.vis_anchor = h

        h = F.average_pooling_2d(h, 7, stride=1)

        if self.uses_original_data:
            # merge data of all 4 individual images in channel dimension
            batch_size, num_channels, height, width = h.shape
            h = F.reshape(h, (batch_size // 4, 4 * num_channels, height, width))

        h = F.relu(self.fc1(h))

        # for each timestep of the localization net do the 'classification'
        h = F.reshape(h, (self.num_timesteps, -1, self.fc1.out_size))
        overall_predictions = []
        for timestep in F.separate(h, axis=0):
            # go 2x num_labels plus 1 timesteps because of ctc loss
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

                lstm_predictions = list(reversed(blstm_predictions))

            if self.use_attention:
                lstm_predictions = self.attend(lstm_predictions)

            final_lstm_predictions = []
            for lstm_prediction in lstm_predictions:
                classified = self.classifier(lstm_prediction)
                final_lstm_predictions.append(F.expand_dims(classified, axis=0))

            final_lstm_predictions = F.concat(final_lstm_predictions, axis=0)
            overall_predictions.append(final_lstm_predictions)

        return overall_predictions, rois, points


class FSNSNet(Chain):

    def __init__(self, localization_net, recognition_net, uses_original_data=False):
        super(FSNSNet, self).__init__()
        with self.init_scope():
            self.localization_net = localization_net
            self.recognition_net = recognition_net

        self._train = True
        self.uses_original_data = uses_original_data

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.localization_net.train = value
        self.recognition_net.train = value

    def __call__(self, images, label=None):
        if self.uses_original_data:
            # handle each individual view as increase in batch size
            batch_size, num_channels, height, width = images.shape
            images = F.reshape(images, (batch_size, num_channels, height, 4, -1))
            images = F.transpose(images, (0, 3, 1, 2, 4))
            images = F.reshape(images, (batch_size * 4, num_channels, height, width // 4))

        batch_size = images.shape[0]
        h = self.localization_net(images)
        new_batch_size = h.shape[0]
        batch_size_increase_factor = new_batch_size // batch_size
        images = F.concat([images for _ in range(batch_size_increase_factor)], axis=0)

        if label is None:
            return self.recognition_net(images, h)
        return self.recognition_net(images, h, label)


class FSNSResnetReuseNet(Chain):

    def __init__(self, target_shape, num_timesteps, num_labels, dropout_ratio=0.5, uses_original_data=False, use_blstm=False):
        super().__init__()
        with self.init_scope():
            self.data_bn = L.BatchNormalization(3)
            self.conv0 = L.Convolution2D(None, 64, 7, stride=2, pad=3, nobias=True)
            self.bn0 = L.BatchNormalization(64)
            self.rs1_1 = ResnetBlock(64)
            self.rs1_2 = ResnetBlock(64)
            self.rs2_1 = ResnetBlock(128, filter_increase=True)
            self.rs2_2 = ResnetBlock(128)
            self.rs3_1 = ResnetBlock(256, filter_increase=True)
            self.rs3_2 = ResnetBlock(256)
            # self.rs4_1 = ResnetBlock(512, filter_increase=True)
            # self.rs4_2 = ResnetBlock(512)

            # localization part
            self.lstm = L.LSTM(None, 256)
            self.transform_2 = L.LSTM(256, 6)

            # recognition part
            self.fc1 = L.Linear(None, 256)
            self.recognition_lstm = L.LSTM(None, 256)
            if use_blstm:
                self.recognition_blstm = L.LSTM(None, 256)
            self.classifier = L.Linear(None, 134)

        self.uses_original_data = uses_original_data
        self.use_blstm = use_blstm
        self.num_timesteps =num_timesteps
        self.num_labels = num_labels
        self.dropout_ratio = dropout_ratio
        self.target_shape = target_shape

        self.localization_vis_anchor = None
        self.recognition_vis_anchor = None

    def localization_net(self, images):
        self.lstm.reset_state()
        self.transform_2.reset_state()

        images = self.data_bn(images)
        h = F.relu(self.bn0(self.conv0(images)))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)

        h = self.rs1_1(h)
        h = self.rs1_2(h)

        h = self.rs2_1(h)
        h = self.rs2_2(h)

        h = self.rs3_1(h)
        h = self.rs3_2(h)

        # h = self.rs4_1(h)
        # h = self.rs4_2(h)

        self.localization_vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        localizations = []

        with cuda.get_device_from_array(h.data):
            for _ in range(self.num_timesteps):
                in_feature = h
                lstm_prediction = F.relu(self.lstm(in_feature))
                transformed = self.transform_2(lstm_prediction)
                transformed = F.reshape(transformed, (-1, 2, 3))
                localizations.append(rotation_dropout(transformed, ratio=self.dropout_ratio))

        return F.concat(localizations, axis=0)

    def recognition_net(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        connected_rois = self.data_bn(rois)
        h = F.relu(self.bn0(self.conv0(connected_rois)))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)

        h = self.rs1_1(h)
        h = self.rs1_2(h)

        h = self.rs2_1(h)
        h = self.rs2_2(h)

        h = self.rs3_1(h)
        h = self.rs3_2(h)

        # h = self.rs4_1(h)
        # h = self.rs4_2(h)

        self.recognition_vis_anchor = h

        h = F.average_pooling_2d(h, 5, stride=1)

        if self.uses_original_data:
            # merge data of all 4 individual images in channel dimension
            batch_size, num_channels, height, width = h.shape
            h = F.reshape(h, (batch_size // 4, 4 * num_channels, height, width))

        h = F.relu(self.fc1(h))

        # for each timestep of the localization net do the 'classification'
        h = F.reshape(h, (self.num_timesteps, -1, self.fc1.out_size))
        overall_predictions = []
        for timestep in F.separate(h, axis=0):
            # go 2x num_labels plus 1 timesteps because of ctc loss
            lstm_predictions = []
            self.recognition_lstm.reset_state()
            if self.use_blstm:
                self.recognition_blstm.reset_state()

            for _ in range(self.num_labels):
                lstm_prediction = self.recognition_lstm(timestep)
                lstm_predictions.append(lstm_prediction)

            if self.use_blstm:
                blstm_predictions = []
                for lstm_prediction in reversed(lstm_predictions):
                    blstm_prediction = self.recognition_blstm(lstm_prediction)
                    blstm_predictions.append(blstm_prediction)

                lstm_predictions = reversed(blstm_predictions)

            final_lstm_predictions = []
            for lstm_prediction in lstm_predictions:
                classified = self.classifier(lstm_prediction)
                final_lstm_predictions.append(F.expand_dims(classified, axis=0))

            final_lstm_predictions = F.concat(final_lstm_predictions, axis=0)
            overall_predictions.append(final_lstm_predictions)

        return overall_predictions, rois, points

    def __call__(self, images):
        if self.uses_original_data:
            # handle each individual view as increase in batch size
            batch_size, num_channels, height, width = images.shape
            images = F.reshape(images, (batch_size, num_channels, height, 4, -1))
            images = F.transpose(images, (0, 3, 1, 2, 4))
            images = F.reshape(images, (batch_size * 4, num_channels, height, width // 4))

        batch_size = images.shape[0]
        localization = self.localization_net(images)
        new_batch_size = localization.shape[0]
        batch_size_increase_factor = new_batch_size // batch_size
        images = F.concat([images for _ in range(batch_size_increase_factor)], axis=0)
        return self.recognition_net(images, localization)
