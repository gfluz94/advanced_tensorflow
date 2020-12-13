import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class MyCategoricalCrossEntropy(Loss):

    def __init__(self, weights=None):
        '''Custom categorical cross entropy designed to account for imbalanced problems

        Attributes:
            weights     Multiplication factor for each class error
        '''
        super().__init__()
        self.weights = weights

    def call(self, y_true, y_pred):
        '''Calculation of the categorical cross entropy error

        Attributes:
            y_true     True labels
            y_pred     Predictions, model's output
        '''
        assert self.weights is None or y_pred.shape[-1]==len(self.weights), "Weights must have the same dimension of the number of classes"
        log_pred = K.log(K.clip(y_pred, K.epsilon(), 1))
        weights = self.weights
        if self.weights is None:
            weights = 1
        sum_example = K.sum(weights*y_true*log_pred, axis=-1)
        return -K.mean(sum_example)