#!/usr/bin/env python3

import argparse
import os
import os.path as path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from mobilenet.datasets import load_imagenet
from mobilenet.models import MobileNet


def train(args):
    model = MobileNet(
        input_size=tuple(args.size), pool_bn_relu=args.pool_bn_relu)

    _ensure_exists(args.checkpoint_dir)

    initial_epoch = 0
    if args.weight_file is not None:
        model.load_weights(args.weight_file)
    else:
        checkpoint = None
        for filename in os.listdir(args.checkpoint_dir):
            base = path.splitext(path.basename(filename))[0]
            pieces = base.split('_')
            if pieces[-1] == 'best':
                continue
            epoch = int(pieces[-1])
            if args.name == '_'.join(pieces[:-1]) and epoch > initial_epoch:
                initial_epoch = epoch
                checkpoint = filename
        if checkpoint is not None:
            model.load_weights(path.join(args.checkpoint_dir, checkpoint))

    train_data, train_steps = load_imagenet('train', tuple(args.size))
    val_data, val_steps = load_imagenet('val', tuple(args.size))

    optimizer_class = getattr(tf.keras.optimizers, args.optimizer)
    if len(args.learning_rates) > 1:
        learning_rate = PiecewiseConstantDecay(
            [x * train_steps for x in args.learning_rate_boundaries],
            args.learning_rates)
    else:
        learning_rate = args.learning_rates[0]
    optimizer = optimizer_class(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    best_filename = path.join(args.checkpoint_dir, args.name + '_best.h5')
    callbacks = [
        ModelCheckpoint(
            path.join(args.checkpoint_dir, args.name + '_{epoch:d}.h5')),
        ModelCheckpoint(best_filename, save_best_only=True)]
    if args.tensorboard_dir != '':
        _ensure_exists(args.tensorboard_dir)
        now_str = datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        callbacks.append(TensorBoard(
            log_dir=path.join(args.tensorboard_dir, args.name + now_str)))

    model.fit(
        x=train_data,
        epochs=args.epochs,
        verbose=args.verbosity,
        callbacks=callbacks,
        validation_data=val_data,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_steps,
        validation_steps=val_steps)

    _ensure_exists(args.model_dir)
    model.load_weights(best_filename)
    model.save(path.join(args.model_dir, args.name + '.h5'))

    for filename in os.listdir(args.checkpoint_dir):
        base = path.splitext(path.basename(filename))[0]
        pieces = base.split('_')
        if args.name == '_'.join(pieces[:-1]):
            os.remove(path.join(args.checkpoint_dir, filename))


def _ensure_exists(dirname):
    if not path.isdir(dirname):
        os.makedirs(dirname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    parser.add_argument(
        'name',
        help='A unique name for the model. Checkpoints will be saved '
             'as <name>_<epoch>.h5 and <name>_best.h5, and the final '
             'model will be saved as <name>.h5.')

    parser.add_argument(
        '-h', '--help', action='help',
        help='Display this help message and exit.')

    parser.add_argument(
        '-C', '--checkpoint-dir', default='checkpoints',
        help='The directory for saving and loading checkpoints.')
    parser.add_argument(
        '-M', '--model-dir', default='models',
        help='The directory where the final model should be saved.')
    parser.add_argument(
        '-T', '--tensorboard-dir', default='tensorboard',
        help='The directory where TensorBoard logs should be written. '
             'TensorBoard logging is disabled if this is an empty '
             'string.')
    parser.add_argument(
        '-w', '--weight-file',
        help='A file from which the model weights should be loaded. '
             'This disables any automatic checkpoint loading.')

    parser.add_argument(
        '-p', '--pool-bn-relu', action='store_true',
        help='Whether to add batch norm and ReLU layers after average '
             'pooling.')
    parser.add_argument(
        '-s', '--size', nargs=2, default=[224, 224], type=int,
        help='The height and width (in that order) to which ImageNet '
             'images should be resized.')

    parser.add_argument(
        '-e', '--epochs', default=50, type=int,
        help='The number of training epochs.')
    parser.add_argument(
        '-l', '--learning-rates', nargs='+', default=[0.001], type=float,
        help='A list of one or more learning rate values.')
    parser.add_argument(
        '-L', '--learning-rate-boundaries', nargs='*', default=[], type=int,
        help='The boundaries (in units of epochs) at which the '
             'learning rate should be changed. Should contain one '
             'fewer value than -l/--learning-rates.')
    parser.add_argument(
        '-o', '--optimizer', default='RMSprop',
        help='The name of the optimizer (case-sensitive). See the '
             'classes listed in the tf.keras.optimizers documentation '
             'for a list of acceptable values.')
    parser.add_argument(
        '-v', '--verbosity', type=int, default=1, choices=[0, 1, 2],
        help='Information to print during training. 0 = silent, '
             '1 = progress bar, 2 = one line per epoch.')

    # This strategy splits batches over the available GPUs
    with tf.distribute.MirroredStrategy().scope():
        train(parser.parse_args())
