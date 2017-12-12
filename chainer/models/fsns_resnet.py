import chainer.functions as F
import chainer.links as L

from chainer import Chain
from chainer.links.model.vision.resnet import ResNetLayers


class FSNSResNetLayers(ResNetLayers):

    @property
    def functions(self):
        functions = super().functions
        del functions['fc6']
        del functions['prob']
        return functions


class FSNSRecognitionResnet(Chain):

    def __init__(self, target_shape, num_labels, num_timesteps, uses_original_data=False, dropout_ratio=0.5, use_dropout=False, use_blstm=False, use_attention=False):
        super().__init__()
        with self.init_scope():
            self.resnet = FSNSResNetLayers('', 152)
            self.fc1 = L.Linear(None, 512)
            self.lstm = L.LSTM(None, 512)
            if use_blstm:
                self.blstm = L.LSTM(None, 512)
            self.classifier = L.Linear(None, 134)

        self.target_shape = target_shape
        self.num_labels = num_labels
        self.num_timesteps = num_timesteps
        self.uses_original_data = uses_original_data
        self.vis_anchor = None
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_blstm = use_blstm
        self.use_attention = use_attention

    def __call__(self, images, localizations):
        points = F.spatial_transformer_grid(localizations, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        h = self.resnet(rois, layers=['res5', 'pool5'])

        self.vis_anchor = h['res5']
        h = h['pool5']

        if self.uses_original_data:
            # merge data of all 4 individual images in channel dimension
            batch_size, num_channels = h.shape
            h = F.reshape(h, (batch_size // 4, 4 * num_channels))

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

            final_lstm_predictions = []
            for lstm_prediction in lstm_predictions:
                classified = self.classifier(lstm_prediction)
                final_lstm_predictions.append(F.expand_dims(classified, axis=0))

            final_lstm_predictions = F.concat(final_lstm_predictions, axis=0)
            overall_predictions.append(final_lstm_predictions)

        return overall_predictions, rois, points
