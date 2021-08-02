#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from glob import glob
import os
# turn off GPU processing because
# tensorflow-gpu can lead to trouble if not installed correctly
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# This script pretrains the neural network. Since it has a similar structure to an
# auto encoder it is here trained like one.
# The architecture is the same but without skip connections, since the neural network
# would in this scenario just send all data directly through the skip connections.

# this data generator streams pairs of two times the same image to the GPU
# because our target is the same as the input
# based on https://www.kaggle.com/mukulkr/camvid-segmentation-using-unet
class DataGenerator(Sequence):

    def __init__(self, pair, batch_size, shuffle):
        self.pair = pair
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(tf.math.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [k for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        self.indexes = tf.range(len(self.pair))
        if self.shuffle == True:
            tf.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        batch_images = []
        batch_masks = []

        for i in list_IDs_temp:
            image = tf.io.read_file(self.pair[i][0])
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.crop_to_bounding_box(image, offset_height=44, offset_width=304, target_height=992,
                                                 target_width=1312)
            # image = tf.image.resize(image, [256, 256]) # for local testing
            batch_images.append(image)
            batch_masks.append(image)

        return tf.stack(batch_images), tf.stack(batch_masks)

def make_pairs(path):
    pairs = []
    image_paths = sorted(glob(os.path.join(path, "pretraining_images/*")))
    # image_paths = sorted(glob(os.path.join(path, "short/2021-04-18_09-34-57 (5-4-2021 12-39-24 PM)/*")))

    for i in range(len(image_paths)):
        pairs.append((image_paths[i], image_paths[i]))

    return pairs


pairs = make_pairs("/home/mlps_team003/CloudChamberProject/")
# pairs = make_pairs("C:/Users/Nicola/Desktop/Uni/MLproject/images")
batch_size = 8
train_generator = DataGenerator(pair=pairs, batch_size=batch_size, shuffle=True)


trainset_length = len(pairs)
steps_per_epoch = trainset_length // batch_size

#define the single convolutional blocks
def conv_block(input, amount_filters, kernel_size):
    x = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal", padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x

def unet(input, amount_filters):
    conv_block1 = conv_block(input, amount_filters, 3) #Encoder
    pooling1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block1)
    dropout1 = tf.keras.layers.Dropout(0.5)(pooling1)


    conv_block2 = conv_block(dropout1, amount_filters * 2, 3)
    pooling2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block2)
    dropout2 = tf.keras.layers.Dropout(0.5)(pooling2)


    conv_block3 = conv_block(dropout2, amount_filters * 4, 3)
    pooling3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block3)
    dropout3 = tf.keras.layers.Dropout(0.5)(pooling3)


    conv_block4 = conv_block(dropout3, amount_filters * 8, 3)
    pooling4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block4)
    dropout4 = tf.keras.layers.Dropout(0.5)(pooling4)


    encoded_features = conv_block(dropout4, amount_filters * 16, 3)


    upsample_block1 = tf.keras.layers.UpSampling2D()(encoded_features) #Decoder
    upsample_block1 = tf.keras.layers.Conv2D(filters=amount_filters * 8, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block1)
    dropout5 = tf.keras.layers.Dropout(0.5)(upsample_block1)
    conv_block5 = conv_block(dropout5, amount_filters * 8, 3)


    upsample_block2 = tf.keras.layers.UpSampling2D()(conv_block5)
    upsample_block2 = tf.keras.layers.Conv2D(filters=amount_filters * 4, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block2)
    dropout6 = tf.keras.layers.Dropout(0.5)(upsample_block2)
    conv_block6 = conv_block(dropout6, amount_filters * 4, 3)


    upsample_block3 = tf.keras.layers.UpSampling2D()(conv_block6)
    upsample_block3 = tf.keras.layers.Conv2D(filters=amount_filters * 2, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block3)
    dropout7 = tf.keras.layers.Dropout(0.5)(upsample_block3)
    conv_block7 = conv_block(dropout7, amount_filters * 2, 3)


    upsample_block4 = tf.keras.layers.UpSampling2D()(conv_block7)
    upsample_block4 = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block4)
    dropout8 = tf.keras.layers.Dropout(0.5)(upsample_block4)
    conv_block8 = conv_block(dropout8, amount_filters, 3)


    output = tf.keras.layers.Conv2D(3, (1, 1), activation="linear")(conv_block8)
    unet = tf.keras.Model(inputs=[input], outputs=[output])
    return unet

input = tf.keras.layers.Input(shape=[992, 1312, 3])
# input = tf.keras.layers.Input(shape=[256, 256, 3]) # for local testing

unet = unet(input, 8)

unet.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

metrics_logger = tf.keras.callbacks.TensorBoard(log_dir="./pretraining_logs", update_freq='epoch', write_images=True)

unet_history = unet.fit(train_generator, epochs=5, steps_per_epoch=steps_per_epoch, callbacks=[metrics_logger])

unet.save("pretraining.h5")
