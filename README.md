# SEE: Towards Semi-Supervised End-to-End Scene Text Recognition
Code for the AAAI 2018 publication "SEE: Towards Semi-Supervised End-to-End Scene Text Recognition". You can read a preprint on [Arxiv](http://arxiv.org/abs/1712.05404)


# Installation

1. Make sure to use Python 3
2. It is a good idea to create a virtual environment ([example for creating a venv](http://docs.python-guide.org/en/latest/dev/virtualenvs/))
3. Make sure you have the latest version of [CUDA](https://developer.nvidia.com/cuda-zone) (>= 8.0) installed
4. Install [CUDNN](https://developer.nvidia.com/cudnn) (> 6.0)
5. Install [NCCL](https://developer.nvidia.com/cudnn) (> 2.0)
6. Install all requirements with the following command: `pip install -r requirements.txt`
7. Check that chainer can use the GPU:
    - start the python interpreter: `python`
    - import chainer: `import chainer`
    - check that cuda is available: `chainer.cuda.available`
    - check that cudnn is enabled: `chainer.cuda.cudnn_enabled`
    - the output of both commands should be `True`


# SVHN Experiments

We performed several experiments on the SVHN dataset.
First, we tried to see whether our architecture is able to reach competitive results
on the SVHN recognition challenge.
Second, we wanted to determine whether our localization network can find text
distributed on a given grid.
In our last experiment we created a dataset, where we randomly distributed
the text samples on the image.

## Datasets

This section describes what needs to be done in order to get/prepare the data.
There is no need for creating the custom datasets by yourself, we also offer them for download.
The information on how to create the datasets is included here for reference.

### Original SVHN data
1. Get the original SVHN datset from [here](http://ufldl.stanford.edu/housenumbers/).
2. Extract the label data using the script `datasets/svn/svhn_dataextract_to_json.py`.
3. use the script `datasets/svhn/prepare_svhn_crops.py` to crop all bounding boxes,
including some background from the SVHN images. Use the script like that:
`python prepare_svhn_crops.py <path to svhn json> 64 <where to save the cropped images> <name of stage>`.
For more information about possible commands you can use `python prepare_svhn_crops.py -h`.

### Grid Dataset
1. Follow steps 1 and 2 of the last subsection in order to get all SVHN images and the corresponding groundtruth.
2. The script `datasets/svhn/create_svhn_dataset_4_images.py` can be used to create the dataset.
3. The command `python create_svhn_dataset_4_images.py -h` shows all available command line options for this script

### Random Dataset
1. Follow steps 1 and 2 of the first subsection in order to get all SVHN images and the corresponding groundtruth.
2. The script `datasets/svhn/create_svhn_dataset.py` can be used to create the dataset.
3. The command `python create_svhn_dataset.py -h` shows all available command line options for this script.

### Dataset Download

You can also download already created datasets [here](https://bartzi.de/research/see).

## Training the model

You can use the script `train_svhn.py` to train a model that can detect and recognize SVHN like text.
The script is tuned to use the custom datasets and should enable you to redo these experiments.

### Preparations

1. Make sure that you have one of the datasets.
2. For training you will need:
    1. the file `svhn_char_map.json` (you can find it in the folder `datasets/svhn`)
    2. the ground truth files of the dataset you want to use
3. Add one line to the beginning of each ground truth file: `<number of house numbers in image> <max number of chars per house number>`
(both values need to be separated by a tab character). If you are using the grid dataset it could look like that: `4    4`.
3. prepare the curriculum specification as a `json` file, by following this template:
    ```
    [
        {
            "train": "<path to train file>",
            "validation": "<path to validation file>"
        }
    ]
    ```
    if you want to train using the curriculum learning strategy, you just need to add
    further dicts to this list.
3. use the script `chainer/train_svhn.py` for training the network.

### Starting the training

The training can be run on GPU or CPU. You can also use multiple GPUs in a data parallel fashion.
In order to specify which GPU to use just add the command line parameter `-g <id of gpu to use>` e.g. `-g 0` for using the first GPU.

You can get a brief explanation of each command line option of the script `train_svhn.py` by running
the script like this: `python train_svhn.py -h`

You will need to specify at least the following parameters:
- `dataset_specification` - this is the path to the `json` file you just created
- `log_dir` - this is the path to directory where the logs shall be saved
- `--char-map ../datasets/svhn/svhn_char_map.json` - path to the char map for mapping classes to labels.
- `--blank-label 0` - indicates that class 0 is the blank label
- `-b <batch-size>` - set the batch size used for training

# FSNS Experiments

In order to see, whether our idea is applicable in practice, we also
did experiments on the FSNS dataset. The FSNS dataset contains
images of French street name signs. The most notable characteristic of this
dataset is, that this dataset does not contain any annotation for text
localization. This fact makes this dataset quite suitable for our method,
as we claim that we can locate and recognize text, even without the corresponding
ground truth for localization.

## Preparing the Dataset

Getting the dataset and making it usable with deep learning frameworks
like Chainer is not an easy task. We provide some scripts that will download
the dataset, convert it from the tensorflow format to single images and
create a ground truth file, that is usable by our train code.

The folder `datasets/fsns` contains all scripts that are necessary for preparing
the dataset. These steps need to be done:

1. use the script `download_fsns.py` for getting the dataset.
You will need to specify a directory, where the data shall be saved.
2. the script `tfrecord_to_image.py` extracts all images and labels from
the downloaded dataset.
3. next, you will need to transform the original ground truth, to the ground truth
format we used for training. Our ground truth format differs, because we
found that it is not possible to train the model, if the word boundaries are not
explicitly given to the model. We therefore transform the line based ground truth
to a word based ground truth. You can use the script `transform_gt.py` for doing that.
You could call the script like that:
`python transform_gt.py <path to original gt> fsns_char_map.json <path to new gt>`.
4. Because of legacy reasons we advice you to use the script `swap_classes.py`.
With this script we will set the class of the blank label to be `0`, as it is defined in
the class to label map `fsns_char_map.json`. You can invoke the script like this:
`python swap_classes.py <gt_file> <output_file_name> 0 133`

## Training the Network

Before you can start training the network, you will need to do the following preparations:

In the last section we already introduced the `transform_gt.py` script.
As we found that it is only possible to train a new model on the FSNS dataset,
when using a curriculum learning strategy, we need to create a learn curriculum
prior to starting the training. You can do this by following these steps:

1. create ground truth files for each step of the curriculum with the `transform_gt.py`
script.
    1. start with a reasonable number of maximum words (2 is a good choice here)
    2. create a ground truth file with all images that contain max. 2 words by using the `transform_gt.py`
    script: `python transform_gt.py <path to downloaded gt> fsns_char_map.json <path to 2 word gt> --max-words 2 --blank-label 0`
    3. Repeat this step with 3 and 4 words (you can also take 5 and 6, too), but make sure
    to only include images with the corresponding amount of words (`--min-words` is the flag to use)
2. Add the path to your files to a `.json` file tht could be called `curriculum.json`
This file works exactly the same as the file discussed in step 3 in the preparations section
for the SVHN experiments.

Once you are done with this, you can actually train the network :tada:

Training the network happens, by using the `train_fsns.py` script.
`python train_fsns.py -h` shows all available command-line options.
This script works very similar to the `train_svhn.py` script

You will need to specify at least the following parameters:
- `dataset_specification` - this is the path to the `json` file you just created
- `log_dir` - this is the path to directory where the logs shall be saved
- `--char-map ../datasets/fsns/fsns_char_map.json` - path to the char map for mapping classes to labels.
- `--blank-label 0` - indicates that class 0 is the blank label
- `-b <batch-size>` - set the batch size used for training


# Pretrained Models

You can download our best performing model on the FSNS dataset and a model
for our SVHN experiments [here](https://bartzi.de/research/see).


# General Notes on Training

This section contains information about things that happen while a network is training.
It includes a description of all data that is being logged and backed up for each train run
and a description of a tool that can be used to inspect the training, while
it is running.

## Contents of the log dir

The code will create a new subdirectory in the log dir, where it puts all
data that is to be logged. The code logs the following pieces of data:
- it creates a backup of the currently used network definition files
- it saves a snapshot of the model at each epoch, or after `snapshot_interval` iterations (default 5000)
- it saves loss and accuracy values at the configured print interval (each time after 100 iterations)
- it will save the prediction of the model on a given, or randomly chosen sample. This visualization
helps with assessing, whether the network is converging or not. It also enables you to inspect the train progress
while the network is training.

## Inspecting the train progress

If you leave the default settings, you can inspect the progress of the
training in real time, by using the script `show_progress.py`. This script
is located in the folder `utils`. You can get all supported command line arguments
with this command: `python show_progress.py -h`. Normally you will want to start
the program like this: `python show_progress.py`. It will open a TK window.
In case the program complains that it is not able to find TK related libraries,
you will need to install them.

## Creating an animation of plotted train steps

The training script contains a little helper that applies the current
state of the model to an image and saves the result of this application
for each iteration (or the way you configure it).

You can use the script `create_video.py` to create an animation out of these images.
In order to use the script, you will need to install ffmpeg (and have the `ffmpeg` command in your path)
and you will need to install imagemagick (and have the `convert` command in your path).
You can then create a video with this command line call:
`python create_video.py <path to directory with images> <path to destination video>`.
You can learn about further command line arguments with `python create_video.py -h`.

# Evaluation

You can evaluate all models (svhn/fsns) with the script `evaluate.py` in the `chainer` directory.

## Usage

You will need a directory containing the following items:
- log_file of the training
- saved model
- network definition files that have been backed up by the train script
- set the gpu to use with `--gpu <id of gpu>`, the code does currently not work on CPU.
- number of labels per timestep (typically max. 5 for SVHN and 21 for FSNS)

### Evaluating a SVHN model

In order to evaluate a SVHN model, you will need to invoke the script like that:
`python evaluate.py svhn <path to dir with specified items> <name of snapshot to evaluate> <path to ground truth file> <path to char map (e.g. svhn_char_map.json)> --target-shape <input shape for recogntion net (e.g. 50,50)> <number of labels per timestep>`

### Evaluating a FSNS model

In order to evaluate a FSNS model, you will need to invoke the script like that:
`python evaluate.py svhn <path to dir with specified items> <name of snapshot to evaluate> <path to ground truth file> <path to char map (e.g. fsns_char_map.json)> <number of labels per timestep>`

### FSNS Demo

In case you only want to see how the model behaves on a given image, you can use the `fsns_demo.py` script.
This script expects a trained model, an image and a char map an prints you the predicted words in the
image + the predicted bounding boxes.
If you download the model provided [here](https://bartzi.de/research/see), you could call the script like this:
`python fsns_demo.py <path to dataset directory> model_35000.npz <path to example image> ../datasets/fsns/fsns_char_map.json`
It should be fairly easy to extend this script to also work with other models. Just have a look at how the different evaulators create the network
and how they extract the characters from the predictions and you should be good to go!


# Citation

If you find this code useful, please cite our paper:

    @article{bartz2017see,
        title={SEE: Towards Semi-Supervised End-to-End Scene Text Recognition},
        author={Bartz, Christian and Yang, Haojin and Meinel, Christoph},
        journal={arXiv preprint arXiv:1712.05404},
        year={2017}
    }

# Notes

If there is anything totally unclear, or not working, please feel free to file an issue.
If you did anything with the code, feel free to file a PR.

