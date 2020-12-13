import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from custom_models import MyCustomVGG16
from custom_losses import MyCategoricalCrossEntropy

def get_mnist_data(rgb_channel=True, scale=True, one_hot_encode=True):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    if rgb_channel:
        X_train = K.expand_dims(X_train, axis=-1)
        X_test = K.expand_dims(X_test, axis=-1)
    if scale:
        X_train = K.cast(X_train, dtype=tf.float32)/255.
        X_test = K.cast(X_test, dtype=tf.float32)/255.
    if one_hot_encode:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    NUM_CLASSES = 10
    X_train, y_train, X_test, y_test = get_mnist_data()

    model = MyCustomVGG16(NUM_CLASSES)
    model.compile(loss=MyCategoricalCrossEntropy(weights=None), optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=10, epochs=5)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Accuracy: {100*np.round(acc, 4):.2f}%")