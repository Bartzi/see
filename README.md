# SEE: Towards Semi-Supervised End-to-End Scene Text Recognition
Code for the AAAI 2018 publication "SEE: Towards Semi-Supervised End-to-End Scene Text Recognition". You can read a preprint on [Arxiv](https://kekse)


# Installation

1. Make sure to use Python 3
2. It is a good idea to create a virtual environment [example for creating a venv](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
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
2. Extract the label data using the script `datasets/svn/svhn_Dataextract_to_json.py`.
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
    1. the file `svhn_char_map.json`
    2. the ground truth files of the dataset you want to use
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

You can get a brief explanation of each command line option of the script `train_svn.py` by running
the script like this: `python train_svhn.py -h`

You will need to specify at least the following parameters:
- `dataset_specification` - this is the path to the `json` file you just created
- `log_dir` - this is the path to directory where the logs shall be saved

### Results of training

The code will create a new subdirectory in the log dir, where it puts all
data that is to be logged. The code logs the following pieces of data:
- it creates a backup of the currently used network definition files
- it saves a snapshot of the model at each epoch, or after `snapshot_interval` iterations (default 5000)
- it saves loss and accuracy values at the configured print interval (each time after 100 iterations)
- it will save the prediction of the model on a given, or randomly chosen sample. This visualization
helps with assessing, whether the network is converging or not. It also enables you to inspect the train progress
while the network is training.

# FSNS Experiments





