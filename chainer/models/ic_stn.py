import chainer.links as L
import chainer.functions as F

from chainer import Chain, cuda, Deserializer

from functions.rotation_droput import rotation_dropout
from models.fsns import ResnetBlock


class InverseCompositionalLocalizationNet(Chain):

    def __init__(self, dropout_ratio, num_timesteps, num_refinement_steps, target_shape, zoom=0.9, do_parameter_refinement=False):
        super().__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.conv0_1 = L.Convolution2D(None, 32, 3, pad=1)
            self.bn0_1 = L.BatchNormalization(32)
            self.rs1 = ResnetBlock(32)
            self.rs2 = ResnetBlock(48, filter_increase=True)
            self.rs3 = ResnetBlock(48)
            self.rs4 = ResnetBlock(32)
            self.rs5 = ResnetBlock(48, filter_increase=True)
            self.lstm = L.LSTM(None, 256)
            self.transform_2 = L.Linear(256, 6)
            self.refinement_transform = L.Linear(2352, 6)

            for transform_param_layer in [self.transform_2, self.refinement_transform]:
                # initialize transform
                transform_param_layer.W.data[...] = 0

                transform_bias = transform_param_layer.b.data
                transform_bias[[0, 4]] = zoom
                transform_bias[[2, 5]] = 0

            self.refinement_transform.b.data[[0, 4]] = 0.1

            self.dropout_ratio = dropout_ratio
            self.num_timesteps = num_timesteps
            self.num_refinement_steps = num_refinement_steps
            self.target_shape = target_shape
            self.do_parameter_refinement = do_parameter_refinement
            self.vis_anchor = None

    def do_transformation_param_refinement_step(self, images, transformation_params):
        transformation_params = self.remove_homogeneous_coordinates(transformation_params)
        points = F.spatial_transformer_grid(transformation_params, self.target_shape)
        rois = F.spatial_transformer_sampler(images, points)

        # rerun parts of the feature extraction for producing a refined version of the transformation params
        h = self.bn0_1(self.conv0_1(rois))
        h = F.average_pooling_2d(F.relu(h), 2, stride=2)

        h = self.rs4(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.rs5(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        transformation_params = self.refinement_transform(h)
        transformation_params = F.reshape(transformation_params, (-1, 2, 3))
        transformation_params = rotation_dropout(transformation_params, ratio=self.dropout_ratio)
        return transformation_params

    def to_homogeneous_coordinates(self, transformation_params):
        batch_size = transformation_params.shape[0]
        transformation_fill = self.xp.zeros((batch_size, 1, 3), dtype=transformation_params.dtype)
        transformation_fill[:, 0, 2] = 1
        transformation_params = F.concat((transformation_params, transformation_fill), axis=1)
        return transformation_params

    def remove_homogeneous_coordinates(self, transformation_params):
        # we can remove homogeneous axis, as it is still 0 0 1
        axes = F.split_axis(transformation_params, 3, axis=1, force_tuple=True)
        return F.concat(axes[:-1], axis=1)

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
        h = F.average_pooling_2d(h, 5)

        localizations = []

        with cuda.get_device_from_array(h.data):

            for _ in range(self.num_timesteps):
                timestep_localizations = []
                in_feature = h
                lstm_prediction = F.relu(self.lstm(in_feature))
                transformed = self.transform_2(lstm_prediction)
                transformed = F.reshape(transformed, (-1, 2, 3))
                transformation_params = rotation_dropout(transformed, ratio=self.dropout_ratio)
                timestep_localizations.append(transformation_params)

                # self.transform_2.disable_update()

                if self.do_parameter_refinement:
                    transformation_params = self.to_homogeneous_coordinates(transformation_params)
                    # refine the transformation parameters
                    for _ in range(self.num_refinement_steps):
                        transformation_deltas = self.do_transformation_param_refinement_step(images, transformation_params)
                        transformation_deltas = self.to_homogeneous_coordinates(transformation_deltas)

                        transformation_params = F.batch_matmul(transformation_params, transformation_deltas)
                        # transformation_params = F.batch_matmul(transformation_deltas, transformation_params)
                        timestep_localizations.append(transformation_params[:, :-1, :])

                localizations.append(timestep_localizations)

        return [F.concat(loc, axis=0) for loc in zip(*localizations)]

    def serialize(self, serializer):
        super().serialize(serializer)
        # only run rest of method if we are deserializing
        if not issubclass(serializer.__class__, Deserializer):
            return

        # if extra transform params are uninitialized we initialize them with the pretrained version of the previous
        # iteration (if there is any)
        # first check if we are loading a pre-trained model
        if not any('conv0' in file for file in serializer.npz.files):
            # nothing to do if we are not loading a pre-trained model
            return
        # no need to do anything if we already trained a model with extra refinement
        if any('bn0_1' in file for file in serializer.npz.files):
            return

        # copy trained params
        params_to_copy = [(self.conv0_1, self.conv0), (self.bn0_1, self.bn0), (self.rs4, self.rs1), (self.rs5, self.rs2)]
        for target, source in params_to_copy:
            target.copyparams(source)


