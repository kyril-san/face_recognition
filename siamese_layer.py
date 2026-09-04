import tensorflow as tf
import keras
from keras import Model, layers
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Flatten, Layer

class SiameseLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

    
