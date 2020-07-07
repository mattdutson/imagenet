# mobilenet

TensorFlow 2 implementation of MobileNet (based on https://arxiv.org/abs/1704.04861)

## ImageNet 2012 Data

Download the training and validation data from [Academic Torrents](https://academictorrents.com/collection/imagenet-2012). Note that ImageNet does permit peer-to-peer distribution of the data provided you first agree to their terms and conditions (see [image-net.org/download-faq](http://image-net.org/download-faq)). Academic Torrents will require you to check a box indicating that you agree to the ImageNet terms and conditions before it allows you to download.

Place the downloaded `.tar` files in the `imagenet` folder. Extract the archives by running the `./extract.sh` Bash script from the `imagenet` folder.

`imagenet/synets.txt` and `imagenet/val_labels.txt` were taken from the public [TensorFlow models repository](https://github.com/tensorflow/models/blob/master/research/inception/inception/data).

## Conda Environment

To create the `mobilenet` environment, run:
```
conda env create -f environment.yml
```
`environment.yml` lists all required Conda and pip packages.

To enable GPU acceleration, instead run:
```
conda env create -f environment_gpu.yml
```
This requires that NVIDIA drivers and CUDA 10.1 be installed (see the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu)).

## Commit Guidelines

Prepend a flag to each commit indicating the type of change which occurred.
Possible values include:

 - `[A]` Addition of new features
 - `[D]` Documentation
 - `[E]` Change to experiment scripts
 - `[F]` Bug fixes
 - `[M]` Miscellaneous or minor modifications
 - `[R]` Refactoring or restructuring
 - `[S]` Style or formatting improvements
 - `[T]` Changes to unit tests
 - `[U]` Update to dependencies

If there are multiple applicable flags, separate them with commas, for example
`[T,R]`.

As a general rule, don't commit experiment scripts until the experiment is completed.
