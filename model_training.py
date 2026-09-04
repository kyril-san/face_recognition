import tensorflow as tf
import keras
from keras import Model, layers
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Flatten, Layer

class Model_Training():
    def embedding(self):
        inp = Input(shape =(100, 100, 3), name ="Input Layer")
        
        x = Conv2D(64, (10, 10), activation='relu')(inp)
        x = MaxPooling2D(64, (2,2), padding="same")(x)
        
        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = MaxPooling2D(64, (2,2), padding="same")(x)
        
        x = Conv2D(128, (4, 4), activation='relu')(x)
        x = MaxPooling2D(64, (2,2), padding="same")(x)
        
        x = Conv2D(256, (4, 4), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(4096, activation="sigmoid")(x)
        
        return Model(inputs= [inp], outputs=x, name="embedding")