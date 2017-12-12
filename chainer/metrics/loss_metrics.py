import itertools
import json

import chainer
from chainer import cuda
import numpy

import chainer.functions as F


class LossMetrics:
    def __init__(self, blank_symbol, char_map, timesteps, image_size, area_loss_factor=0, aspect_ratio_loss_factor=0, uses_original_data=False,
                 area_scaling_factor=2):
        self.aspect_ratio_loss_factor = aspect_ratio_loss_factor
        self.blank_symbol = blank_symbol
        self.xp = None
        with open(char_map, 'r') as the_char_map:
            self.char_map = json.load(the_char_map)
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.base_area_loss_factor = area_loss_factor
        self.area_scaling_factor = area_scaling_factor
        self.uses_original_data = uses_original_data
        self.area_loss_factor = self.base_area_loss_factor

    def get_label_lengths(self, labels):
        if self.xp == numpy:
            label_lengths = self.xp.zeros(len(labels))

            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels.data[i][j] == self.blank_symbol:
                        label_lengths[i] = j
                        break
        else:
            import cupy
            label_length_kernel = cupy.ElementwiseKernel(
                'raw T labels, int32 blank_symbol, int32 num_labels',
                'T length',
                '''
                    for (int j = 0; j < num_labels; ++j) {
                        T label_value = labels[i * num_labels + j];
                        if (label_value == blank_symbol) {
                            length = j;
                            break;
                        }
                    }
                ''',
                'get_label_lengths'
            )
            label_lengths = label_length_kernel(labels.data, self.blank_symbol, labels.shape[1], size=len(labels))
        return label_lengths

    def strip_prediction(self, predictions):
        # TODO Parallelize
        words = []
        for prediction in predictions:
            stripped_prediction = self.xp.empty((0,), dtype=self.xp.int32)
            for char in prediction:
                if char == self.blank_symbol:
                    continue
                stripped_prediction = self.xp.hstack((stripped_prediction, char.reshape(1,)))
            words.append(stripped_prediction)
        return words

    def get_bbox_side_lengths(self, grids):
        x0, x1, x2, y0, y1, y2 = self.get_corners(grids)

        width = F.sqrt(
            F.square(x1 - x0) + F.square(y1 - y0)
        )

        height = F.sqrt(
            F.square(x2 - x0) + F.square(y2 - y0)
        )
        return width, height

    def get_corners(self, grids):
        _, _, height, width = grids.shape
        grids = (grids + 1) / 2
        x_points = grids[:, 0, ...] * self.image_size.width
        y_points = grids[:, 1, ...] * self.image_size.height
        top_left_x = F.get_item(x_points, [..., 0, 0])
        top_left_y = F.get_item(y_points, [..., 0, 0])
        top_right_x = F.get_item(x_points, [..., 0, width - 1])
        top_right_y = F.get_item(y_points, [..., 0, width - 1])
        bottom_left_x = F.get_item(x_points, [..., height - 1, 0])
        bottom_left_y = F.get_item(y_points, [..., height - 1, 0])
        return top_left_x, top_right_x, bottom_left_x, top_left_y, top_right_y, bottom_left_y

    def calc_direction_loss(self, grids):
        top_left_x, top_right_x, _, top_left_y, _, bottom_left_y = self.get_corners(grids)

        # penalize upside down images
        distance = top_left_y - bottom_left_y
        loss_values = F.maximum(distance, self.xp.zeros_like(distance))
        up_down_loss = F.average(loss_values)

        # penalize images that are vertically mirrored
        distance = top_left_x - top_right_x
        loss_values = F.maximum(distance, self.xp.zeros_like(distance))
        left_right_loss = F.average(loss_values)

        return up_down_loss + left_right_loss

    def calc_height_loss(self, height):
        # penalize bboxes that are not high enough to contain text (10 pixels)
        shifted_height = height - 10
        thresholded_height = F.minimum(shifted_height, self.xp.zeros_like(shifted_height))
        thresholded_height *= -1

        return F.average(thresholded_height)

    def calc_area_loss(self, width, height):
        loc_area = width * height
        loc_ratio = loc_area / (self.image_size.width * self.image_size.height)
        return sum(loc_ratio) / max(len(loc_ratio), 1)

    def calc_overlap(self, left_1, width_1, left_2, width_2):
        radius_1 = width_1 / 2
        center_1 = left_1 + radius_1
        radius_2 = width_2 / 2
        center_2 = left_2 + radius_2

        center_distance = center_2 - center_1
        center_distance = F.maximum(center_distance, center_distance * -1)
        min_distance_for_no_overlap = radius_1 + radius_2
        return min_distance_for_no_overlap - center_distance

    def calc_intersection(self, top_left_x_1, width_1, top_left_x_2, width_2, top_left_y_1, height_1, top_left_y_2, height_2):
        width_overlap = self.calc_overlap(
            top_left_x_1,
            width_1,
            top_left_x_2,
            width_2
        )

        height_overlap = self.calc_overlap(
            top_left_y_1,
            height_1,
            top_left_y_2,
            height_2
        )

        width_overlap = F.maximum(width_overlap, self.xp.zeros_like(width_overlap))
        height_overlap = F.maximum(height_overlap, self.xp.zeros_like(height_overlap))

        return width_overlap * height_overlap

    def calc_iou_loss(self, grids1, grids2):
        top_left_x_1, top_right_x_1, _, top_left_y_1, _, bottom_left_y_1 = self.get_corners(grids1)
        top_left_x_2, top_right_x_2, _, top_left_y_2, _, bottom_left_y_2 = self.get_corners(grids2)

        width_1 = top_right_x_1 - top_left_x_1
        width_2 = top_right_x_2 - top_left_x_2
        height_1 = bottom_left_y_1 - top_left_y_1
        height_2 = bottom_left_y_2 - top_left_y_2
        intersection = self.calc_intersection(top_left_x_1, width_1, top_left_x_2, width_2, top_left_y_1, height_1, top_left_y_2, height_2)
        union = width_1 * height_1 + width_2 * height_2 - intersection
        iou = intersection / F.maximum(union, self.xp.ones_like(union))

        return sum(iou) / len(iou)

    def calc_aspect_ratio_loss(self, width, height, label_lengths=None):
        # penalize aspect ratios that are higher than wide, and penalize aspect ratios that are tooo wide
        aspect_ratio = height / F.maximum(width, self.xp.ones_like(width))
        # do not give an incentive to bboxes with a width that is 2x the height of the box
        aspect_loss = F.maximum(aspect_ratio - 0.5, self.xp.zeros_like(aspect_ratio))

        # penalize very long bboxes (based on the underlying word), by assuming that a single letter
        # has a max width of its height, if the width of the bbox is too large it will be penalized
        if label_lengths is not None:
            max_width = label_lengths * height
            width_ratio = width - max_width
            width_threshold = F.maximum(width_ratio, self.xp.zeros_like(width_ratio))
            aspect_loss = aspect_ratio + width_threshold

        return sum(aspect_loss) / len(aspect_loss)

    def label_to_char(self, label):
        return chr(self.char_map[str(label)])

    def calc_loss(self, x, t):
        batch_predictions, _, grids = x
        self.xp = cuda.get_array_module(batch_predictions[0], t)

        # reshape labels
        batch_size = t.shape[0]
        t = F.reshape(t, (batch_size, self.num_timesteps, -1))

        # reshape grids
        grid_shape = grids.shape
        if self.uses_original_data:
            grids = F.reshape(grids, (self.num_timesteps, batch_size, 4,) + grid_shape[1:])
        else:
            grids = F.reshape(grids, (self.num_timesteps, batch_size, 1,) + grid_shape[1:])
        losses = []

        # with cuda.get_device_from_array(grids.data):
        #     grid_list = F.separate(F.reshape(grids, (self.num_timesteps, -1,) + grids.shape[3:]), axis=0)
        #     overlap_losses = []
        #     for grid_1, grid_2 in itertools.combinations(grid_list, 2):
        #         overlap_losses.append(self.calc_iou_loss(grid_1, grid_2))
        #     losses.append(sum(overlap_losses) / max(len(overlap_losses), 1))

        loss_weights = [1, 1.25, 2, 1.25]
        for i, (predictions, grid, labels) in enumerate(zip(batch_predictions, F.separate(grids, axis=0), F.separate(t, axis=1)), start=1):
            with cuda.get_device_from_array(getattr(predictions, 'data', predictions[0].data)):
                # adapt ctc weight depending on current prediction position and labels
                # if all labels are blank, we want this weight to be full weight!
                overall_loss_weight = loss_weights[i - 1]
                loss = self.calc_actual_loss(predictions, grid, labels)
                # label_lengths = self.get_label_lengths(labels)

                for sub_grid in F.separate(grid, axis=1):
                    width, height = self.get_bbox_side_lengths(sub_grid)
                    loss += self.area_loss_factor * self.calc_area_loss(width, height)
                    loss += self.aspect_ratio_loss_factor * self.calc_aspect_ratio_loss(width, height)
                    loss += self.calc_direction_loss(sub_grid)
                    loss += self.calc_height_loss(height)
                loss *= overall_loss_weight
                losses.append(loss)

        return sum(losses) / len(losses)

    def calc_actual_loss(self, predictions, grid, labels):
        raise NotImplementedError

    def scale_area_loss_factor(self, accuracy):
        self.area_loss_factor = self.base_area_loss_factor + self.area_scaling_factor * accuracy

    def calc_accuracy(self, x, t):
        batch_predictions, _, _ = x
        self.xp = cuda.get_array_module(batch_predictions[0], t)
        batch_size = t.shape[0]
        t = F.reshape(t, (batch_size, self.num_timesteps, -1))
        accuracies = []

        for predictions, labels in zip(batch_predictions, F.separate(t, axis=1)):
            if isinstance(predictions, list):
                predictions = F.concat([F.expand_dims(p, axis=0) for p in predictions], axis=0)
            with cuda.get_device_from_array(predictions.data):

                classification = F.softmax(predictions, axis=2)
                classification = classification.data
                classification = self.xp.argmax(classification, axis=2)
                classification = self.xp.transpose(classification, (1, 0))

                words = self.strip_prediction(classification)
                labels = self.strip_prediction(labels.data)

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
