# mobilenet

TensorFlow 2 implementation of MobileNet (based on https://arxiv.org/abs/1704.04861) 

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
