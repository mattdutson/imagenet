import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 32
N_CLASSES = 1000
N_PREFETCH = 64
N_VAL = 100000


def load_imagenet(split, size=(224, 224)):
    def _map(item):
        image = tf.image.convert_image_dtype(item['image'], tf.float32)
        image = tf.image.resize_with_pad(image, size[0], size[1])
        label = tf.one_hot(item['label'], N_CLASSES)
        return image, label

    def _prepare(dataset):
        return (dataset
                .map(_map)
                .repeat()
                .prefetch(N_PREFETCH)
                .batch(BATCH_SIZE))

    if split in ('train', 'val'):
        full, info = tfds.load('imagenet2012', split='train', with_info=True)
        n_train = info.splits['train'].num_examples - N_VAL
        if split == 'train':
            return _prepare(full.take(n_train)), _n_batches(n_train)
        else:
            return _prepare(full.skip(n_train)), _n_batches(N_VAL)
    elif split == 'test':
        test, info = tfds.load('imagenet2012', split='val', with_info=True)
        n_test = info.splits['val'].num_examples
        return _prepare(test), _n_batches(n_test)
    else:
        raise ValueError('split must be "train", "val", or "test"')


def _n_batches(n_items):
    return int(np.ceil(n_items / BATCH_SIZE))
