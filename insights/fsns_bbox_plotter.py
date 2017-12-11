import chainer.functions as F
from PIL import Image

from insights.bbox_plotter import BBOXPlotter
from utils.datatypes import Size


class FSNSBBOXPlotter(BBOXPlotter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = Size(height=self.image.shape[1], width=self.image.shape[2] // 4)

    @property
    def original_image_paste_location(self):
        return self.image_size.width, 0

    def get_num_timesteps(self, bboxes):
        return bboxes.shape[0] // 4

    def set_output_sizes(self, backprop_vis, bboxes, image, num_timesteps):
        _, num_channels, height, width = bboxes.shape
        image_height = image.height
        if len(backprop_vis) > 0:
            image_height = image.height + self.image_size.height
        if self.render_extracted_rois:
            image_height = image.height + self.image_size.height * (1 + num_timesteps)

        dest_image = Image.new("RGBA", (image.width + self.image_size.width, image_height), color='black')
        bboxes = F.reshape(bboxes, (num_timesteps, 4, num_channels, height, self.out_size.width))

        return bboxes, dest_image

    def show_backprop_vis(self, backprop_vis, dest_image, image, num_timesteps):
        # first render localization visualization
        for j, vis in enumerate(backprop_vis[0]):
            backprop_image = self.array_to_image(self.xp.tile(vis, (3, 1, 1)))
            dest_image.paste(backprop_image, ((j + 1) * self.image_size.width, image.height))
        # second render recognition visualization
        _, num_channels, height, width = backprop_vis[1].shape
        recognition_vis = self.xp.reshape(backprop_vis[1], (num_timesteps, -1, num_channels, height, width))
        for i in range(len(recognition_vis)):
            for j, vis in enumerate(recognition_vis[i]):
                backprop_image = self.array_to_image(self.xp.tile(vis, (3, 1, 1)))\
                    .resize((self.image_size.width, self.image_size.height))
                dest_image.paste(backprop_image, ((j + 1) * self.image_size.width, (i + 2) * image.height))

    def render_extracted_regions(self, dest_image, image, rois, num_timesteps):
        _, num_channels, height, width = rois.shape
        rois = self.xp.reshape(rois, (num_timesteps, -1, num_channels, height, width))

        for i, roi in enumerate(rois, start=1):
            roi_image = self.variable_to_image(roi[0])
            paste_location = 0, (i + 1) * image.height
            dest_image.paste(roi_image.resize((self.image_size.width, self.image_size.height)), paste_location)
