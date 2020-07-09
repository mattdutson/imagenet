from tensorflow.keras import Sequential
from tensorflow.keras.layers import *


class MobileNet(Sequential):
    def __init__(
            self,
            input_size=(224, 224),
            final_relu=False,
            pool_bn_relu=False,
            n_classes=1000):
        super(MobileNet, self).__init__()

        self.add(Conv2D(
            32, (3, 3),
            strides=2,
            padding='same',
            input_shape=input_size + (3,)))
        self._add_bn_relu()
        self._add_depthwise_block()
        self._add_pointwise_block(64)

        self._add_depthwise_block(strides=2)
        self._add_pointwise_block(128)
        self._add_depthwise_block()
        self._add_pointwise_block(128)

        self._add_depthwise_block(strides=2)
        self._add_pointwise_block(256)
        self._add_depthwise_block()
        self._add_pointwise_block(256)

        self._add_depthwise_block(strides=2)
        self._add_pointwise_block(512)
        for _ in range(5):
            self._add_depthwise_block()
            self._add_pointwise_block(512)

        self._add_depthwise_block(strides=2)
        self._add_pointwise_block(1024)
        self._add_depthwise_block()
        self._add_pointwise_block(1024)

        self.add(AveragePooling2D(
            pool_size=(input_size[0] // 32, input_size[1] // 32)))
        if pool_bn_relu:
            self._add_bn_relu()

        self.add(Flatten())
        self.add(Dense(n_classes))
        if final_relu:
            self.add(ReLU())
        self.add(Softmax())

    def _add_bn_relu(self):
        self.add(BatchNormalization(scale=False))
        self.add(ReLU())

    def _add_depthwise_block(self, strides=1):
        self.add(DepthwiseConv2D((3, 3), strides=strides, padding='same'))
        self._add_bn_relu()

    def _add_pointwise_block(self, filters):
        self.add(Conv2D(filters, (1, 1)))
        self._add_bn_relu()
