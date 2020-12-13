import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Model

from custom_layers import MyDenseLayer


class MyCustomVGG16(Model):

    def __init__(self, num_classes):
        '''Custom VGG model encapsulated in one class to provide organization within application scripts

        Attributes:
            num_classes     Number of classes in classification task
        '''
        super().__init__()

        conv_filters = [64, 128, 256, 512]
        kernel_sizes = [3, 3, 3, 3]
        repetitions = [2, 2, 3, 3]
        self.blocks = [Block(f, k, r) for f, k, r in zip(conv_filters, kernel_sizes, repetitions)]
        self.flatten = Flatten()
        self.fc_dense = MyDenseLayer(256, activation="relu")
        self.softmax = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.fc_dense(x)
        return self.softmax(x)


class Block(Model):

    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        '''Convolution followed by MaxPooling block that is repeated across VGG sctructure

        Attributes:
            filters         Number of filters in convolutional layer
            kernel_size     Kernel size to be applied in convolutional layer
            repetitions     Number of times concolution layer is repeated sequentially
            pool_size       Pool size in max pooling layer
            strides         Number os strides in max pooling layer
        '''
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        self.convs2D = [Conv2D(self.filters, self.kernel_size, activation="relu", padding="same") for _ in range(self.repetitions)]
        self.max_pool = MaxPool2D(pool_size=pool_size, strides=strides)

    def call(self, inputs):
        x = inputs
        for conv_layer in self.convs2D:
            x = conv_layer(x)
        return self.max_pool(x)


