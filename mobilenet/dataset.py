import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 32
N_CLASSES = 1000
N_PREFETCH = 64
N_VAL = 100000

BRIGHTNESS_DELTA = 0.1
CONTRAST_MIN = 0.8
CONTRAST_MAX = 1.0 / 0.8
HUE_DELTA = 0.05
SATURATION_MIN = 0.8
SATURATION_MAX = 1.0 / 0.8


def imagenet(split, size=(320, 320), augment=False):
    """
    Loads a split of the ImageNet dataset.

    :param string split: Should be 'train', 'test', or 'val'.
    :param tuple size: The height and width (in that order) to which
        images should be resized.
    :param bool augment: Whether to apply data augmentation.
    :return (tensorflow.data.Dataset, int): The dataset split and number
        of batches in the split.
    """

    def preprocess(item):
        # float32 is better behaved during augmentation
        x = tf.image.convert_image_dtype(item['image'], tf.float32)

        if augment:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, BRIGHTNESS_DELTA)
            x = tf.image.random_contrast(x, CONTRAST_MIN, CONTRAST_MAX)
            x = tf.image.random_hue(x, HUE_DELTA)
            x = tf.image.random_saturation(x, SATURATION_MIN, SATURATION_MAX)

            # Brightness and contrast shifts may put pixel values out of
            # the range [0, 1]
            x = tf.clip_by_value(x, 0.0, 1.0)

        # Pad after augmentation so zero padded pixels aren't changed
        x = tf.image.resize_with_pad(x, size[0], size[1])
        y = tf.one_hot(item['label'], N_CLASSES)
        return x, y

    def prepare(dataset):
        return (
            dataset
                .map(preprocess)
                .repeat()
                .prefetch(N_PREFETCH)
                .batch(BATCH_SIZE))

    if split in ['train', 'val']:
        full, info = tfds.load('imagenet2012', split='train', with_info=True)
        n_train = info.splits['train'].num_examples - N_VAL
        if split == 'train':
            return prepare(full.take(n_train)), _n_batches(n_train)
        else:
            return prepare(full.skip(n_train)), _n_batches(N_VAL)
    elif split == 'test':
        test, info = tfds.load('imagenet2012', split='val', with_info=True)
        n_test = info.splits['val'].num_examples
        return prepare(test), _n_batches(n_test)
    else:
        raise ValueError('split must be "train", "val", or "test"')


def _n_batches(n_items):
    return int(np.ceil(n_items / BATCH_SIZE))
