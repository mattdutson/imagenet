#!/usr/bin/env python3

import argparse
import os
import os.path as path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from mobilenet.dataset import imagenet
from mobilenet.model import mobilenet


def main(args):
    tf.random.set_seed(0)

    # Search for an existing checkpoint
    _ensure_exists(args.checkpoint_dir)
    initial_epoch = 0
    checkpoint = None
    for filename in os.listdir(args.checkpoint_dir):
        pieces = path.splitext(path.basename(filename))[0].split('_')
        if pieces[-1] == 'best':
            continue
        epoch = int(pieces[-1])
        if args.name == '_'.join(pieces[:-1]) and epoch > initial_epoch:
            initial_epoch = epoch
            checkpoint = filename

    # Load or create the model
    if checkpoint is not None:
        model = load_model(path.join(args.checkpoint_dir, checkpoint))
        need_compile = False
        need_learning_rate = False
    elif args.model_file is not None:
        model = load_model(args.model_file)
        need_compile = False
        need_learning_rate = True
    else:
        model = mobilenet(input_size=tuple(args.size), l2_decay=args.l2_decay)
        if args.weight_file is not None:
            model.load_weights(args.weight_file)
        need_compile = True
        need_learning_rate = True

    # Load the dataset (train and validation splits)
    train_data, train_steps = imagenet('train', tuple(args.size), augment=not args.no_augment)
    val_data, val_steps = imagenet('val', tuple(args.size), augment=False)

    # Prepare the model for training
    if need_compile:
        model.compile(
            optimizer=getattr(tf.keras.optimizers, args.optimizer)(),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy'])
    if need_learning_rate:
        if len(args.learning_rates) > 1:
            model.optimizer.learning_rate = PiecewiseConstantDecay(
                [x * train_steps for x in args.learning_rate_boundaries],
                args.learning_rates)
        else:
            model.optimizer.learning_rate = args.learning_rates[0]

    # Set up training callbacks (checkpointing and TensorBoard)
    best_filename = path.join(args.checkpoint_dir, args.name + '_best.h5')
    callbacks = [
        ModelCheckpoint(path.join(args.checkpoint_dir, args.name + '_{epoch:d}.h5')),
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

    # Save the model with the lowest validation loss
    _ensure_exists(args.model_dir)
    model = load_model(best_filename)
    model.save(path.join(args.model_dir, args.name + '.h5'))

    # Remove checkpoints if training completes successfully
    for filename in os.listdir(args.checkpoint_dir):
        pieces = path.splitext(path.basename(filename))[0].split('_')
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
        '-m', '--model-file',
        help='A file from which the entire model (including the '
             'optimizer state) should be loaded. This is ignored when '
             'training is automatically resumed from a checkpoint.')
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
             'This is ignored when training is automatically resumed '
             'from a checkpoint.')

    parser.add_argument(
        '-A', '--no-augment', action='store_true',
        help='Do not apply data augmentation to training images.')
    parser.add_argument(
        '-d', '--l2-decay', default=1e-3, type=float,
        help='The amount of L2 weight decay to add to the loss. '
             'Applies only to kernels, not biases.')
    parser.add_argument(
        '-e', '--epochs', default=75, type=int,
        help='The number of training epochs.')
    parser.add_argument(
        '-l', '--learning-rates',
        nargs='+', default=[1e-2, 1e-3, 1e-4], type=float,
        help='A list of one or more learning rate values.')
    parser.add_argument(
        '-L', '--learning-rate-boundaries',
        nargs='*', default=[25, 50], type=int,
        help='The boundaries (in units of epochs) at which the '
             'learning rate should be changed. Should contain one '
             'fewer value than -l/--learning-rates.')
    parser.add_argument(
        '-o', '--optimizer', default='SGD',
        help='The name of the optimizer (case-sensitive). See the '
             'classes listed in the tf.keras.optimizers documentation '
             'for a list of acceptable values.')
    parser.add_argument(
        '-s', '--size', nargs=2, default=[320, 320], type=int,
        help='The height and width (in that order) to which images '
             'should be resized.')
    parser.add_argument(
        '-v', '--verbosity', default=2, type=int, choices=[0, 1, 2],
        help='Information to print during training. 0 = silent, 1 = '
             'progress bar, 2 = one line per epoch.')

    # This strategy splits batches over the available GPUs
    with tf.distribute.MirroredStrategy().scope():
        main(parser.parse_args())
