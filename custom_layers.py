import tensorflow as tf
from tensorflow.keras.layers import Layer


class MyDenseLayer(Layer):

    def __init__(self, units, activation=None):
        '''Custom dense layer which has a quadratic additional term: a*x**2 + b*x + c

        Attributes:
            units     Number of units in the layer
            activation  Activation function to be applied to layer's output
        '''
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        '''Initializing weights randomly.

        Attributes:
            input_shape     Input shape from previous layer
        '''
        squared_init = tf.random_normal_initializer()
        linear_init = tf.random_normal_initializer()
        bias_init = tf.zeros_initializer()

        self.squared_coeff = tf.Variable(name="quadratic coefficients",
                                         initial_value=squared_init(shape=(input_shape[-1], self.units), dtype="float32"),
                                         trainable=True)
        self.linear_coeff = tf.Variable(name="linear coefficients",
                                        initial_value=linear_init(shape=(input_shape[-1], self.units), dtype="float32"),
                                        trainable=True)
        self.bias = tf.Variable(name="linear bias",
                                initial_value=bias_init(shape=(self.units,), dtype="float32"),
                                trainable=True)

    def call(self, inputs):
        '''Performing matricial operations: a*x**2 + b*x + c

        Attributes:
            inputs     Input from previous layer
        '''
        squared_term = tf.matmul(tf.square(inputs), self.squared_coeff)
        linear_term = tf.matmul(inputs, self.linear_coeff)
        total = squared_term+linear_term+self.bias
        return self.activation(total)
