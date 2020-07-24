#!/usr/bin/env python3

import argparse
import os.path as path

import numpy as np
import tensorflow as tf

from mobilenet.datasets import load_imagenet
from mobilenet.utils import ensure_exists


def save_examples(args):
    ensure_exists(args.examples_dir)

    data, _ = load_imagenet(args.split, tuple(args.size))
    for i, item in enumerate(data.unbatch()):
        if i >= args.n_examples:
            break

        class_id = np.argmax(item[1])
        filename = path.join(
            args.examples_dir, '{}_{}.jpeg'.format(i, class_id))
        image_uint8 = tf.image.convert_image_dtype(item[0], tf.uint8)
        tf.io.write_file(filename, tf.io.encode_jpeg(image_uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    parser.add_argument(
        '-h', '--help', action='help',
        help='Display this help message and exit.')

    parser.add_argument(
        '-E', '--examples-dir', default='examples',
        help='Examples are saved to a subdirectory of this directory '
             'corresponding to their split ("train", "val", or '
             '"test").')

    parser.add_argument(
        '-n', '--n-examples', default=20, type=int,
        help='The number of examples to save.')
    parser.add_argument(
        '-p', '--split', default='train', choices=['train', 'val', 'test'],
        help='The dataset split from which examples should be pulled.')
    parser.add_argument(
        '-s', '--size', nargs=2, default=[224, 224], type=int,
        help='The height and width (in that order) to which images '
             'should be resized.')

    save_examples(parser.parse_args())
