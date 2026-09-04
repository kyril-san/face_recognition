import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from siamese_layer import SiameseLayer
import keras
from keras import Model, layers
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Flatten, Layer
from sklearn.metrics import classification_report
from model_training import Model_Training
import matplotlib.pyplot as plt


def preprocess(filepath):
    byte_img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img /255.0
    return img

# def preprocess_twin(input_image, validation_image, label):
#     return (preprocess(input_image), preprocess(validation_image), label) 
def preprocess_twin(input_image, validation_image, label):
    return {
        "Input_img": preprocess(input_image),
        "validation_img": preprocess(validation_image)
    }, label
    
def map_data(Data):
    data = Data.map(preprocess_twin)
    data = data.shuffle(buffer_size = 1024)
    test_data = data.skip(round(len(data) * .7))
    test_data = test_data.take(round(len(data)* .3)).cache()
    test_data = test_data.batch(8)
    test_data = test_data.prefetch(16)
    
    train_data = data.take(round(len(data) * .7)).cache()
    train_data = train_data.batch(8)
    train_data = train_data.prefetch(16)
    return train_data, test_data

def map_data2(Data):
    Data = Data.map(preprocess_twin).cache()
    Data = Data.shuffle(buffer_size=10000000, seed=42, reshuffle_each_iteration=False)
    total_size = round(len(Data))
    train_size = round(total_size * 0.7)
    test_valid_size = total_size - train_size
    test_size = round(test_valid_size * 0.5)
    
    train_data = Data.take(train_size)
    test_valid_data = Data.skip(train_size)
    test_data = test_valid_data.take(test_size)
    valid_data = test_valid_data.skip(test_size)
    
    train_data = train_data.batch(16).prefetch(tf.data.AUTOTUNE)
    
    valid_data = valid_data.batch(16).prefetch(tf.data.AUTOTUNE)
    
    test_data = test_data.batch(16).prefetch(tf.data.AUTOTUNE)
    
    return train_data, test_data, valid_data
    
def make_siamese_model():
        embeddings = Model_Training().embedding()
        embeddings.summary()
        input_image = Input(name= 'Input_img', shape=(100, 100, 3))
        validation_image = Input(name= 'validation_img', shape=(100, 100, 3))
        inp_embedding = embeddings(input_image)
        val_embedding = embeddings(validation_image)
        
        siamese_layer = SiameseLayer()
        siamese_layer._name = "distance"
        distances = siamese_layer(inp_embedding,val_embedding)
        
        classifier = Dense(1,activation="sigmoid")(distances)
        
        return Model(inputs=[input_image, validation_image], outputs=classifier, name= 'Siamese_Models')

def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

if __name__ == "__main__":
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"        # disable XLA's op-level JIT clustering
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"    # disable the GPU autotuner (the thing that was JIT-compiling alt kernel configs since your very first error in this thread)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"    
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # tf.config.optimizer.set_jit(False)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    Neg_path = os.path.join("data", "Neg_img")
    Anch_path = os.path.join("data", "Anchor_img")
    Pos_path = os.path.join("data", "Pos_img")
    best_checkpoint = os.path.join('checkpoints', 'best')
    weight_checkpoint = os.path.join('checkpoints', 'weights')
    model_checkpoint = os.path.join('checkpoints', 'model')
    length = 10000
    
    Neg = tf.data.Dataset.list_files(os.path.join(Neg_path, "*.jpg")).take(length)
    Anchor = tf.data.Dataset.list_files(os.path.join(Anch_path, "*.jpg")).take(length)
    Pos = tf.data.Dataset.list_files(os.path.join(Pos_path, "*.jpg")).take(length)

    
    print(f"Neg : {len(Neg)} \nPos : {len(Pos)} \nAnchor: {len(Anchor)}")

    Positives = tf.data.Dataset.zip((Anchor, Pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(Anchor)))))
    Negatives = tf.data.Dataset.zip((Anchor, Neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(Anchor)))))
    
    data = Positives.concatenate(Negatives)
    
    train_data, test_data, valid_data = map_data2(data)
    model = make_siamese_model()
    model.summary()

    model.compile(optimizer= keras.optimizers.AdamW(learning_rate=1e-5), loss = tf.losses.BinaryCrossentropy() ,metrics=['accuracy']
        , jit_compile=False)
    history = model.fit(train_data, validation_data= valid_data,  batch_size=16, epochs=15, callbacks=[
        # keras.callbacks.EarlyStopping(
        #     patience=3, monitor="val_loss", restore_best_weights=True
            
        # ), 
        # keras.callbacks.ModelCheckpoint(
        #     filepath=(f"{best_checkpoint}/best6.keras"),
        #     monitor="val_loss",
        #     save_best_only=True
        # )
    ])
    
    results = model.evaluate(test_data)
    print(results)
    model.save(f"{model_checkpoint}/modelV8.keras")
    model.save_weights(f"{weight_checkpoint}/modelv8.weights.h5")
    
    # visualize the testing done
    for input_img, label in test_data.take(1):
        predicted = []
        for i, (input_image, validation_image) in enumerate(zip(input_img['Input_img'], input_img['validation_img'])):
            predictions = model.predict({
                "Input_img": tf.expand_dims(input_image, axis=0),
                "validation_img": tf.expand_dims(validation_image, axis=0)
            })
            predicted.append(predictions)
            plt.subplot(1,2,1)
            plt.title(f'input imagez\n Actual Label {label[i]}')
            
            plt.imshow(input_image)
            
            plt.subplot(1,2,2)
            plt.title(f'Validation Image\n Predicted Label {predictions}')
            plt.imshow(validation_image)
            plt.show()
    print(f"labels  is {label}\nPredictions is {predicted}")