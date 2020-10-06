#!/usr/bin/env python3

import argparse
import os
import os.path as path

import numpy as np
import tensorflow as tf

from mobilenet.dataset import imagenet


def main(args):
    tf.random.set_seed(0)
    if not path.isdir(args.examples_dir):
        os.makedirs(args.examples_dir)

    data, _ = imagenet(args.split, tuple(args.size), augment=args.augment)
    for i, item in enumerate(data.unbatch()):
        if i >= args.n_examples:
            break
        class_id = np.argmax(item[1])
        filename = path.join(args.examples_dir, '{}_{}.jpeg'.format(i, class_id))
        image_uint8 = tf.image.convert_image_dtype(item[0], tf.uint8)
        tf.io.write_file(filename, tf.io.encode_jpeg(image_uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    parser.add_argument(
        '-h', '--help', action='help',
        help='Display this help message and exit.')

    parser.add_argument(
        '-E', '--examples-dir', default='examples',
        help='Examples are saved to a subdirectory of this directory '
             'corresponding to their split ("train", "val", or '
             '"test").')

    parser.add_argument(
        '-a', '--augment', action='store_true',
        help='Apply data augmentation. During actual training this '
             'should only be done with training-set images. Here it '
             'can be done with any split.')
    parser.add_argument(
        '-n', '--n-examples', default=10, type=int,
        help='The number of examples to save.')
    parser.add_argument(
        '-s', '--size', nargs=2, default=[320, 320], type=int,
        help='The height and width (in that order) to which images '
             'should be resized.')
    parser.add_argument(
        '-S', '--split', default='train', choices=['train', 'val', 'test'],
        help='The dataset split from which examples should be pulled.')

    main(parser.parse_args())
