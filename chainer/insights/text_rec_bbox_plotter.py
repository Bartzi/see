from chainer import cuda

import chainer.functions as F
from PIL import Image, ImageDraw

from insights.bbox_plotter import BBOXPlotter


class TextRecBBOXPlotter(BBOXPlotter):

    def __init__(self, *args, **kwargs):
        self.render_intermediate_bboxes = kwargs.pop('render_intermediate_bboxes', False)
        super().__init__(*args, **kwargs)

    def get_num_timesteps(self, bboxes):
        return bboxes[-1].shape[0]

    def set_output_sizes(self, backprop_vis, bboxes, image, num_timesteps):
        _, num_channels, height, width = bboxes[-1].shape

        image_height = image.height if len(backprop_vis) == 0 else image.height + self.image_size.height
        image_width = image.width + image.width * num_timesteps if self.render_extracted_rois else image.width

        dest_image = Image.new("RGBA", (image_width, image_height), color='black')
        bboxes = F.concat([F.reshape(bbox, (num_timesteps, 1, num_channels, height, width)) for bbox in bboxes], axis=1)

        return bboxes, dest_image

    def render_extracted_regions(self, dest_image, image, rois, num_timesteps):
        rois = rois[-1]
        _, num_channels, height, width = rois.shape
        rois = self.xp.reshape(rois, (num_timesteps, -1, num_channels, height, width))

        for i, roi in enumerate(rois, start=1):
            roi_image = self.variable_to_image(roi[0])
            paste_location = i * image.width, 0
            dest_image.paste(roi_image.resize((self.image_size.width, self.image_size.height)), paste_location)

    def decode_predictions(self, predictions):
        # concat all individual predictions and slice for each time step
        predictions = F.concat([F.expand_dims(prediction, axis=0) for prediction in predictions], axis=0)

        with cuda.get_device_from_array(predictions.data):
            prediction = F.squeeze(predictions, axis=1)
            classification = F.softmax(prediction, axis=1)
            classification = classification.data
            classification = self.xp.argmax(classification, axis=1)

            words = self.loss_metrics.strip_prediction(classification[self.xp.newaxis, ...])[0]
            word = "".join(map(self.loss_metrics.label_to_char, words))

        return word

    def draw_bboxes(self, bboxes, image):
        draw = ImageDraw.Draw(image)
        for boxes, colour in zip(F.separate(bboxes, axis=0), self.colours):
            num_boxes = boxes.shape[0]

            for i, bbox in enumerate(F.separate(boxes, axis=0)):
                # render all intermediate results with lower alpha as the others
                fill_colour = colour
                if i < num_boxes - 1:
                    if not self.render_intermediate_bboxes:
                        continue
                    fill_colour += '88'

                bbox.data[...] = (bbox.data[...] + 1) / 2
                bbox.data[0, :] *= self.image_size.width
                bbox.data[1, :] *= self.image_size.height

                x = self.xp.clip(bbox.data[0, :].reshape(self.out_size), 0, self.image_size.width)
                y = self.xp.clip(bbox.data[1, :].reshape(self.out_size), 0, self.image_size.height)

                top_left = (x[0, 0], y[0, 0])
                top_right = (x[0, -1], y[0, -1])
                bottom_left = (x[-1, 0], y[-1, 0])
                bottom_right = (x[-1, -1], y[-1, -1])

                corners = [top_left, top_right, bottom_right, bottom_left]
                next_corners = corners[1:] + [corners[0]]

                for first_corner, next_corner in zip(corners, next_corners):
                    draw.line([first_corner, next_corner], fill=fill_colour, width=3)
