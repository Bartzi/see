import base64
import json
import os
import socket

from io import BytesIO

import chainer.functions as F
import numpy as np
from PIL import Image, ImageFont

from PIL import ImageDraw

import chainer
from chainer import cuda
from chainer.training import Extension

from insights.visual_backprop import VisualBackprop
from utils.datatypes import Size


COLOR_MAP = [
    "#00B3FF",  # Vivid Yellow
    "#753E80",  # Strong Purple
    "#0068FF",  # Vivid Orange
    "#D7BDA6",  # Very Light Blue
    "#2000C1",  # Vivid Red
    "#62A2CE",  # Grayish Yellow
    "#667081",  # Medium Gray

    # The following don't work well for people with defective color vision
    "#347D00",  # Vivid Green
    "#8E76F6",  # Strong Purplish Pink
    "#8A5300",  # Strong Blue
    "#5C7AFF",  # Strong Yellowish Pink
    "#7A3753",  # Strong Violet
    "#008EFF",  # Vivid Orange Yellow
    "#5128B3",  # Strong Purplish Red
    "#00C8F4",  # Vivid Greenish Yellow
    "#0D187F",  # Strong Reddish Brown
    "#00AA93",  # Vivid Yellowish Green
    "#153359",  # Deep Yellowish Brown
    "#133AF1",  # Vivid Reddish Orange
    "#162C23",  # Dark Olive Green

    # extend colour map
    "#00B3FF",  # Vivid Yellow
    "#753E80",  # Strong Purple
    "#0068FF",  # Vivid Orange
    "#D7BDA6",  # Very Light Blue
    "#2000C1",  # Vivid Red
    "#62A2CE",  # Grayish Yellow
    "#667081",  # Medium Gray
]


class BBOXPlotter(Extension):

    def __init__(self, image, out_dir, out_size, loss_metrics, **kwargs):
        super(BBOXPlotter, self).__init__()
        self.image = image
        self.render_extracted_rois = kwargs.pop("render_extracted_rois", True)
        self.image_size = Size(height=image.shape[1], width=image.shape[2])
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_size = out_size
        self.colours = COLOR_MAP
        self.send_bboxes = kwargs.pop("send_bboxes", False)
        self.upstream_ip = kwargs.pop("upstream_ip", '127.0.0.1')
        self.upstream_port = kwargs.pop("upstream_port", 1337)
        self.loss_metrics = loss_metrics
        self.font = ImageFont.truetype("utils/DejaVuSans.ttf", 20)
        self.visualization_anchors = kwargs.pop("visualization_anchors", [])
        self.visual_backprop = VisualBackprop()
        self.xp = np

    def send_image(self, data):
        height = data.height
        width = data.width
        channels = len(data.getbands())

        # convert image to png in order to save network bandwidth
        png_stream = BytesIO()
        data.save(png_stream, format="PNG")
        png_stream = png_stream.getvalue()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.upstream_ip, self.upstream_port))
            except Exception as e:
                print(e)
                print("could not connect to display server, disabling image rendering")
                self.send_bboxes = False
                return
            data = {
                'width': width,
                'height': height,
                'channels': channels,
                'image': base64.b64encode(png_stream).decode('utf-8'),
            }
            sock.send(bytes(json.dumps(data), 'utf-8'))

    def array_to_image(self, array):
        if array.shape[0] == 1:
            # image is black and white, we need to trick the system into thinking, that we are having an RGB image
            array = self.xp.tile(array, (3, 1, 1))
        return Image.fromarray(cuda.to_cpu(array.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB").convert("RGBA")

    def variable_to_image(self, variable):
        return self.array_to_image(variable.data)

    def __call__(self, trainer):
        iteration = trainer.updater.iteration

        with cuda.get_device_from_id(trainer.updater.get_optimizer('main').target._device_id), chainer.using_config('train', False):
            self.xp = np if trainer.updater.get_optimizer('main').target._device_id < 0 else cuda.cupy
            image = self.xp.asarray(self.image)
            predictor = trainer.updater.get_optimizer('main').target.predictor
            predictions, rois, bboxes = predictor(image[self.xp.newaxis, ...])

            backprop_visualizations = []
            for visanchor in self.visualization_anchors:
                vis_target = predictor
                for target in visanchor:
                    vis_target = getattr(vis_target, target)
                backprop_visualizations.append(self.visual_backprop.perform_visual_backprop(vis_target))

            self.render_rois(predictions, rois, bboxes, iteration, self.image.copy(), backprop_vis=backprop_visualizations)

    @property
    def original_image_paste_location(self):
        return 0, 0

    def render_rois(self, predictions, rois, bboxes, iteration, image, backprop_vis=()):
        # get the predicted text
        text = self.decode_predictions(predictions)

        image = self.array_to_image(image)

        num_timesteps = self.get_num_timesteps(bboxes)
        bboxes, dest_image = self.set_output_sizes(backprop_vis, bboxes, image, num_timesteps)
        if self.render_extracted_rois:
            self.render_extracted_regions(dest_image, image, rois, num_timesteps)

        if len(backprop_vis) != 0:
            # if we have a backprop visualization we can show it now
            self.show_backprop_vis(backprop_vis, dest_image, image, num_timesteps)

        self.draw_bboxes(bboxes, image)
        dest_image.paste(image, self.original_image_paste_location)
        if len(text) > 0:
            dest_image = self.render_text(dest_image, text)
        dest_image.save("{}.png".format(os.path.join(self.out_dir, str(iteration))), 'png')
        if self.send_bboxes:
            self.send_image(dest_image)

    def get_num_timesteps(self, bboxes):
        return bboxes.shape[0]

    def set_output_sizes(self, backprop_vis, bboxes, image, num_timesteps):
        _, num_channels, height, width = bboxes.shape

        image_height = image.height if len(backprop_vis) == 0 else image.height + self.image_size.height
        image_width = image.width + image.width * num_timesteps if self.render_extracted_rois else image.width

        dest_image = Image.new("RGBA", (image_width, image_height), color='black')
        bboxes = F.reshape(bboxes, (num_timesteps, 1, num_channels, height, width))

        return bboxes, dest_image

    def show_backprop_vis(self, backprop_vis, dest_image, image, num_timesteps):
        count = 0
        for visualization in backprop_vis:
            for vis in visualization:
                backprop_image = self.array_to_image(self.xp.tile(vis[0], (3, 1, 1))).resize(
                    (self.image_size.width, self.image_size.height))
                dest_image.paste(backprop_image, (count * backprop_image.width, image.height))
                count += 1

    def decode_predictions(self, predictions):
        words = []
        for prediction in predictions:
            if isinstance(prediction, list):
                prediction = F.concat([F.expand_dims(p, axis=0) for p in prediction], axis=0)

            prediction = self.xp.transpose(prediction.data, (1, 0, 2))
            prediction = self.xp.squeeze(prediction, axis=0)
            prediction = self.xp.argmax(prediction, axis=1)
            word = self.loss_metrics.strip_prediction(prediction[self.xp.newaxis, ...])[0]
            if len(word) == 1 and word[0] == 0:
                continue
            word = "".join(map(self.loss_metrics.label_to_char, word))
            word = word.replace(chr(self.loss_metrics.char_map[str(self.loss_metrics.blank_symbol)]), '')
            if len(word) > 0:
                words.append(word)
        text = " ".join(words)
        return text

    def render_extracted_regions(self, dest_image, image, rois, num_timesteps):
        _, num_channels, height, width = rois.shape
        rois = self.xp.reshape(rois, (num_timesteps, -1, num_channels, height, width))

        for i, roi in enumerate(rois, start=1):
            roi_image = self.variable_to_image(roi[0])
            paste_location = i * image.width, 0
            dest_image.paste(roi_image.resize((self.image_size.width, self.image_size.height)), paste_location)

    def render_text(self, dest_image, text):
        label_image = Image.new(dest_image.mode, dest_image.size)
        # only keep ascii characters
        # labels = ''.join(filter(lambda x: len(x) == len(x.encode()), labels))
        draw = ImageDraw.Draw(label_image)
        text_width, text_height = draw.textsize(text, font=self.font)
        draw.rectangle([dest_image.width - text_width - 1, 0, dest_image.width, text_height],
                       fill=(255, 255, 255, 160))
        draw.text((dest_image.width - text_width - 1, 0), text, fill='green', font=self.font)
        dest_image = Image.alpha_composite(dest_image, label_image)
        return dest_image

    def draw_bboxes(self, bboxes, image):
        draw = ImageDraw.Draw(image)
        for i, sub_box in enumerate(F.separate(bboxes, axis=1)):
            for bbox, colour in zip(F.separate(sub_box, axis=0), self.colours):
                bbox.data[...] = (bbox.data[...] + 1) / 2
                bbox.data[0, :] *= self.image_size.width
                bbox.data[1, :] *= self.image_size.height

                x = self.xp.clip(bbox.data[0, :].reshape(self.out_size), 0, self.image_size.width) + i * self.image_size.width
                y = self.xp.clip(bbox.data[1, :].reshape(self.out_size), 0, self.image_size.height)

                top_left = (x[0, 0], y[0, 0])
                top_right = (x[0, -1], y[0, -1])
                bottom_left = (x[-1, 0], y[-1, 0])
                bottom_right = (x[-1, -1], y[-1, -1])

                corners = [top_left, top_right, bottom_right, bottom_left]
                next_corners = corners[1:] + [corners[0]]

                for first_corner, next_corner in zip(corners, next_corners):
                    draw.line([first_corner, next_corner], fill=colour, width=3)
