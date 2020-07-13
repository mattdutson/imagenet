from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2


class MobileNet(Sequential):
    def __init__(
            self,
            input_size=(224, 224),
            l2_decay=0.0,
            n_classes=1000):
        super(MobileNet, self).__init__()
        self.l2_decay = l2_decay

        self.add(Conv2D(
            32, (3, 3),
            strides=2,
            padding='same',
            kernel_regularizer=l2(l=self.l2_decay),
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

        self.add(Flatten())
        self.add(Dense(n_classes, kernel_regularizer=l2(l=self.l2_decay)))
        self.add(Softmax())

    def _add_bn_relu(self):
        self.add(BatchNormalization(scale=False))
        self.add(ReLU())

    def _add_depthwise_block(self, strides=1):
        self.add(DepthwiseConv2D(
            (3, 3),
            strides=strides,
            padding='same',
            kernel_regularizer=l2(l=self.l2_decay)))
        self._add_bn_relu()

    def _add_pointwise_block(self, filters):
        self.add(Conv2D(
            filters, (1, 1),
            kernel_regularizer=l2(l=self.l2_decay)))
        self._add_bn_relu()
