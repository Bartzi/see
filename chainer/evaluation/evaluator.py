import csv
import json
import importlib
import statistics
from itertools import zip_longest

import os

import chainer
import chainer.functions as F
import numpy as np
import tqdm

from chainer import configuration, cuda
from PIL import Image

from insights.bbox_plotter import BBOXPlotter
from insights.fsns_bbox_plotter import FSNSBBOXPlotter
from insights.svhn_bbox_plotter import SVHNBBoxPlotter
from insights.text_rec_bbox_plotter import TextRecBBOXPlotter
from metrics.softmax_metrics import SoftmaxMetrics
from metrics.textrec_metrics import TextRecSoftmaxMetrics
from utils.datatypes import Size


class Evaluator:

    def __init__(self, args):
        self.args = args

        with open(os.path.join(args.model_dir, args.log_name)) as the_log:
            log_data = json.load(the_log)[0]

        self.target_shape = Size._make(log_data['target_size'])
        self.image_size = Size._make(log_data['image_size'])

        # Step 1: build network
        localization_net_class_name, localization_module_name = self.get_class_and_module(log_data['localization_net'])
        module = self.load_module(os.path.abspath(os.path.join(args.model_dir, localization_module_name)))
        localization_net_class = eval('module.{}'.format(localization_net_class_name))
        localization_net = self.build_localization_net(localization_net_class)

        recognition_net_class_name, recognition_module_name = self.get_class_and_module(log_data['recognition_net'])
        module = self.load_module(os.path.abspath(os.path.join(args.model_dir, recognition_module_name)))
        recognition_net_class = eval('module.{}'.format(recognition_net_class_name))
        recognition_net = self.build_recognition_net(recognition_net_class)

        fusion_net_class_name, fusion_module_name = self.get_class_and_module(log_data['fusion_net'])
        module = self.load_module(os.path.abspath(os.path.join(args.model_dir, fusion_module_name)))
        fusion_net_class = eval('module.{}'.format(fusion_net_class_name))
        self.net = self.build_fusion_net(fusion_net_class, localization_net, recognition_net)

        if args.gpu >= 0:
            self.net.to_gpu(args.gpu)

        # Step 2: load weights
        with np.load(os.path.join(args.model_dir, args.snapshot_name)) as f:
            chainer.serializers.NpzDeserializer(f).load(self.net)

        # Step 3: open gt and do evaluation
        with open(args.char_map) as the_map:
            self.char_map = json.load(the_map)

        self.xp = chainer.cuda.cupy if args.gpu >= 0 else np

        with open(args.eval_gt) as eval_gt:
            reader = csv.reader(eval_gt, delimiter='\t')
            self.lines = [l for l in reader]

        self.blank_symbol = args.blank_symbol

        self.num_correct_lines = 0
        self.num_correct_words = 0
        self.num_lines = 0
        self.num_words = 0
        self.num_word_x_correct = [0 for _ in range(args.timesteps)]
        self.num_word_x = [0 for _ in range(args.timesteps)]

        self.model_dir = args.model_dir

        self.metrics = self.create_metrics()

        self.save_rois = args.save_rois
        self.bbox_plotter = None
        if self.save_rois:
            self.create_bbox_plotter()

    def create_bbox_plotter(self):
        raise NotImplementedError

    def build_localization_net(self, localization_net_class):
        raise NotImplementedError

    def build_recognition_net(self, recognition_net_class):
        raise NotImplementedError

    def build_fusion_net(self, fusion_net_class, localization_net, recognition_net):
        raise NotImplementedError

    @staticmethod
    def get_class_and_module(log_data):
        raise NotImplementedError

    def evaluate(self):
        results = []
        with chainer.cuda.get_device(self.args.gpu):
            for i, line in enumerate(tqdm.tqdm(self.lines)):
                image_file = line[0]
                labels = self.prepare_label(line[1:])
                image = self.load_image(image_file)
                with configuration.using_config('train', False):
                    predictions, crops, grids = self.net(image[self.xp.newaxis, ...])

                words, gt_words = self.calc_accuracy(predictions, labels)
                results.append([words, gt_words])

                if self.save_rois:
                    image = self.xp.asarray(image)
                    self.bbox_plotter.xp = self.xp
                    self.bbox_plotter.render_rois(predictions, crops, grids, i, image)

        self.print_results()

        with open(os.path.join(self.model_dir, "eval_results.csv"), "w") as results_file:
            writer = csv.writer(results_file, delimiter=',')
            writer.writerows(results)

    def print_results(self):
            raise NotImplementedError

    def load_module(self, module_file):
        module_spec = importlib.util.spec_from_file_location("models.model", module_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def load_image(self, image_file):
        with Image.open(image_file) as the_image:
            # the_image = the_image.resize((self.image_size.width, self.image_size.height), Image.LANCZOS)
            image = self.xp.asarray(the_image, dtype=np.float32)
            image /= 255
            image = image.transpose(2, 0, 1)
            return image

    def label_to_char(self, label):
        return chr(self.char_map[str(label)])

    def prepare_label(self, data):
        self.xp.array(data, dtype=self.xp.int32)
        return data.reshape((-1, self.args.num_labels))

    def strip_prediction(self, predictions):
        # TODO Parallelize
        words = []
        for prediction in predictions:
            stripped_prediction = self.xp.empty((0,), dtype=self.xp.int32)
            for char in prediction:
                if char == self.blank_symbol:
                    continue
                stripped_prediction = self.xp.hstack((stripped_prediction, char.reshape(1, )))
            words.append(stripped_prediction)
        return words

    def calc_accuracy(self, predictions, labels):
        raise NotImplementedError

    def create_metrics(self):
        raise NotImplementedError


class FSNSEvaluator(Evaluator):

    def create_metrics(self):
        return SoftmaxMetrics(
            self.args.blank_symbol,
            self.args.char_map,
            self.args.timesteps,
            self.image_size,
            uses_original_data=self.args.is_original_fsns,
        )

    def build_recognition_net(self, recognition_net_class):
        return recognition_net_class(
            self.target_shape,
            self.args.num_labels,
            self.args.timesteps,
            uses_original_data=self.args.is_original_fsns,
            use_blstm=True
        )

    def build_localization_net(self, localization_net_class):
        return localization_net_class(self.args.dropout_ratio, self.args.timesteps)

    def build_fusion_net(self, fusion_net_class, localization_net, recognition_net):
        return fusion_net_class(localization_net, recognition_net, uses_original_data=self.args.is_original_fsns)

    def print_results(self):
        print("Sequence Accuracy: {}".format(self.num_correct_lines / self.num_lines))
        print("Word Accuracy: {}".format(self.num_correct_words / self.num_words))
        print("Single word accuracies:")
        for i, (c, n) in enumerate(zip(self.num_word_x_correct, self.num_word_x), start=1):
            print("Accuracy for Word {}: {}".format(i, c / n))

    @staticmethod
    def get_class_and_module(log_data):
        if not isinstance(log_data, list):
            module_name = 'fsns.py'
            klass_name = log_data
        else:
            klass_name, module_name = log_data
        return klass_name, module_name

    def create_bbox_plotter(self):
        BBOXPlotterClass = BBOXPlotter if not self.args.is_original_fsns else FSNSBBOXPlotter
        self.bbox_plotter = BBOXPlotterClass(
            self.load_image(self.lines[0][0]),
            os.path.join(self.args.model_dir, "eval_bboxes"),
            self.target_shape,
            self
        )
        self.lines = self.lines[:self.args.num_rois]

    def calc_accuracy(self, predictions, labels):
        has_error = False
        words = []
        gt_words = []
        for i, (prediction, label) in enumerate(zip_longest(predictions, labels, fillvalue=self.xp.zeros_like(predictions[0]))):
            classification = F.softmax(prediction, axis=2)
            classification = classification.data
            classification = self.xp.argmax(classification, axis=2)
            classification = self.xp.transpose(classification, (1, 0))

            word = self.strip_prediction(classification)[0]
            label = self.strip_prediction(label[self.xp.newaxis, ...])[0]

            word = "".join(map(lambda x: chr(self.char_map[str(x)]), word))
            label = "".join(map(lambda x: chr(self.char_map[str(x)]), label))
            words.append(word)
            gt_words.append(label)

            if word == label:
                self.num_correct_words += 1
                if i < self.args.timesteps:
                    self.num_word_x_correct[i] += 1
            else:
                has_error = True
            self.num_words += 1
            if i < self.args.timesteps:
                self.num_word_x[i] += 1

        if not has_error:
            self.num_correct_lines += 1
        self.num_lines += 1

        return " ".join(words), " ".join(gt_words)


class SVHNEvaluator(Evaluator):

    def create_metrics(self):
        return TextRecSoftmaxMetrics(
            self.args.blank_symbol,
            self.args.char_map,
            self.args.timesteps,
            self.image_size,
        )

    def build_recognition_net(self, recognition_net_class):
        return recognition_net_class(
            self.target_shape,
            self.args.num_labels,
            self.args.timesteps,
            use_blstm=False
        )

    def build_localization_net(self, localization_net_class):
        return localization_net_class(self.args.dropout_ratio, self.args.timesteps)

    def build_fusion_net(self, fusion_net_class, localization_net, recognition_net):
        return fusion_net_class(localization_net, recognition_net)

    def calc_accuracy(self, predictions, labels):
        batch_predictions = predictions
        # concat all individual predictions and slice for each time step
        batch_predictions = F.concat([F.expand_dims(p, axis=2) for p in batch_predictions], axis=2)

        t = F.reshape(labels, (1, self.args.timesteps, -1))

        accuracies = []
        with cuda.get_device_from_array(batch_predictions.data):
            for prediction, label in zip(F.separate(batch_predictions, axis=0), F.separate(t, axis=2)):
                classification = F.softmax(prediction, axis=2)
                classification = classification.data
                classification = self.xp.argmax(classification, axis=2)
                # classification = self.xp.transpose(classification, (1, 0))

                words = self.strip_prediction(classification)
                labels = self.strip_prediction(label.data)

                for word, label in zip(words, labels):
                    word = "".join(map(self.label_to_char, word))
                    label = "".join(map(self.label_to_char, label))
                    if word == label:
                        self.num_correct_words += 1
                    self.num_words += 1

        return word, label

    def print_results(self):
        print("Word Accuracy: {}".format(self.num_correct_words / self.num_words))

    def create_bbox_plotter(self):
        self.bbox_plotter = SVHNBBoxPlotter(
            self.load_image(self.lines[0][0]),
            os.path.join(self.args.model_dir, "eval_bboxes"),
            self.target_shape,
            self
        )
        self.lines = self.lines[:self.args.num_rois]

    @staticmethod
    def get_class_and_module(log_data):
        if not isinstance(log_data, list):
            module_name = 'svhn.py'
            klass_name = log_data
        else:
            klass_name, module_name = log_data
        return klass_name, module_name


class TextRecognitionEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracies = []
        self.reverse_char_map = {v: k for k, v in self.char_map.items()}

    def calc_accuracy(self, predictions, labels):
        self.accuracies.append(self.metrics.calc_accuracy((predictions, None, None), labels))
        #TODO strip prediction and label and return word
        return '', ''

    def print_results(self):
        print("Accuracy: {}".format(statistics.mean(self.accuracies)))

    def create_metrics(self):
        return TextRecSoftmaxMetrics(
            self.args.blank_symbol,
            self.args.char_map,
            self.args.timesteps,
            self.image_size
        )

    def create_bbox_plotter(self):
        self.bbox_plotter = TextRecBBOXPlotter(
            self.load_image(self.lines[0][0]),
            os.path.join(self.args.model_dir, 'eval_bboxes'),
            self.target_shape,
            self.metrics,
            visualization_anchors=[["localization_net", "vis_anchor"], ["recognition_net", "vis_anchor"]],
            render_extracted_rois=False,
            invoke_before_training=True,
            render_intermediate_bboxes=self.args.render_all_bboxes,
        )
        self.lines = self.lines[:self.args.num_rois]

    def build_fusion_net(self, fusion_net_class, localization_net, recognition_net):
        return fusion_net_class(localization_net, recognition_net)

    def build_recognition_net(self, recognition_net_class):
        return recognition_net_class(
            self.target_shape,
            num_rois=self.args.timesteps,
            label_size=52,
        )

    def build_localization_net(self, localization_net_class):
        return localization_net_class(
            self.args.dropout_ratio,
            self.args.timesteps,
            self.args.refinement_steps,
            self.target_shape,
            zoom=1.0,
            do_parameter_refinement=self.args.refinement
        )

    @staticmethod
    def get_class_and_module(log_data):
        if not isinstance(log_data, list):
            if 'InverseCompositional' in log_data:
                module_name = 'ic_stn.py'
                klass_name = log_data
            else:
                module_name = 'text_recognition.py'
                klass_name = log_data
        else:
            klass_name, module_name = log_data
        return klass_name, module_name

    def prepare_label(self, data):
        labels = [int(self.reverse_char_map[ord(character)]) for character in data[0].lower()]
        labels += [self.args.blank_symbol] * (self.args.timesteps - len(labels))
        return self.xp.array(labels, dtype=self.xp.int32)[self.xp.newaxis, ...]

    def load_image(self, image_file):
        with Image.open(image_file) as the_image:
            the_image = the_image.convert('L')
            the_image = the_image.resize((self.image_size.width, self.image_size.height), Image.LANCZOS)
            image = self.xp.asarray(the_image, dtype=np.float32)
            image /= 255
            image = self.xp.broadcast_to(image, (3, self.image_size.height, self.image_size.width))
            return image
