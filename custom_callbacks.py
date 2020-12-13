import matplotlib.pyplot as plt
import numpy as np
import datetime
import io
from PIL import Image
from IPython.display import Image as IPyImage
import imageio

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

GIF_PATH = './animation.gif'


class VisCallback(Callback):
    def __init__(self, inputs, ground_truth, display_freq=10, n_samples=10):
        '''Custom CallBack designed to visualize predictions x ground truth in mnist dataset

        Attributes:
            inputs      input array
            ground_truth       true labels
            display_freq        number of epochs after which a checkpoint image is displayed
            n_samples       how many examples will be sampled and then displayed per epoch
        '''
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples

    def on_epoch_end(self, epoch, logs=None):
        '''Building visualization on epoch end

        Attributes:
            epoch      epoch number
            logs        if user wants to account for logs
        '''
        min_slice = np.random.choice(np.arange(len(self.inputs)-10))
        X_test, y_test = self.inputs[min_slice:min_slice+10, ...], self.ground_truth[min_slice:min_slice+10, ...]
        true_labels = np.argmax(y_test, axis=1)
        predictions = np.argmax(self.model.predict(X_test), axis=1)

        display_digits(X_test, predictions, true_labels, epoch, n=10)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave(GIF_PATH, self.images, fps=1)


plt.rc('font', size=20)
plt.rc('figure', figsize=(15, 3))

def display_digits(inputs, outputs, ground_truth, epoch, n=10):
    plt.clf()
    plt.yticks([])
    plt.grid(None)
    inputs = np.reshape(inputs, [n, 28, 28])
    inputs = np.swapaxes(inputs, 0, 1)
    inputs = np.reshape(inputs, [28, 28*n])
    plt.imshow(inputs, cmap="gray")
    plt.xticks([28*x+14 for x in range(n)], outputs)
    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if outputs[i] == ground_truth[i]: 
            t.set_color('green') 
        else: 
            t.set_color('red')
    plt.grid(None)