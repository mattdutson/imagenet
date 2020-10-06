#!/usr/bin/env python3

import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model

from mobilenet.dataset import imagenet


def test(args):
    model = load_model(args.model_file)
    data, steps = imagenet('test', tuple(args.size))
    model.evaluate(x=data, steps=steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    parser.add_argument(
        'model-file',
        help='The filename (.h5 file) of the model to test.')

    parser.add_argument(
        '-h', '--help', action='help',
        help='Display this help message and exit.')

    parser.add_argument(
        '-s', '--size', nargs=2, default=[320, 320], type=int,
        help='The height and width (in that order) to which images '
             'should be resized.')

    # This strategy splits batches over the available GPUs
    with tf.distribute.MirroredStrategy().scope():
        test(parser.parse_args())
