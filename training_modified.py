#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from glob import glob
import os
from datetime import datetime
# turn off GPU processing because
# tensorflow-gpu can lead to trouble if not installed correctly
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


######## This script is a copy of training.py and enables training on our local GPU in order for us to properly
######## use tensorboard which didn't work on the server and introduce a validation set. It therefore modifies
######## the data generator to only load half the augmented images in one batch and the rest later. There is also
######## a ValDataGenerator that is used for the validation and test data where no augmentation is desired.

# This script trains the neural network.
# First it defines the data pipeline that loads the images and masks and preprocesses them.
# Then it defines the custom loss function.
# Afterwards the architecture of the neural network is defined and lastly the training procedure.



########## DATA PIPELINE ##########


# turns a RGB mask into 4 2D binary class maps where each cell has value 1 if
# the corresponding pixel in the original pixel had the color of the class
def to_one_hot(mask):
    # these are the RGB colors that the masks use, if these values or the colors
    # in the mask images change every class map will be filled with zeros!
    colors = [
        [227, 26, 28],  # Red
        [65, 117, 5],  # Green
        [106, 61, 154],  # Violet
        [31, 120, 180]  # Blue
    ]

    one_hot_mask = []
    for color in colors:
        # tf.equal compares every pixel of the mask with the current RGB color
        # and returns a matrix where a cell is TRUE if the corresponding pixel had the color of the current class
        # reduce all then turns this boolean matrix into a 2D map
        class_mask = tf.reduce_all(tf.equal(mask, color), axis=-1)
        class_mask = tf.cast(class_mask, tf.float32)
        one_hot_mask.append(class_mask)
    # after we have all the 2D class maps we stack on top of each other to get one tensor
    one_hot_encoded_mask = tf.stack(one_hot_mask, axis=-1)

    return one_hot_encoded_mask

# parses the masks, they are converted to float 32 later in to_one_hot
def parse_mask(name):
    mask = tf.io.read_file(name)
    mask = tf.image.decode_png(mask)
    # this cuts the borders off, height and width need to be multiples of 32!
    # mask = tf.image.resize(mask, [256, 256]) # for local testing
    mask = tf.image.crop_to_bounding_box(mask, offset_height=44, offset_width=304, target_height=992, target_width=1312)

    return mask

# parses the images
def parse_image(name):
    image = tf.io.read_file(name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32) # Neural Nets work with float 32
    #image = tf.image.rgb_to_grayscale(image)
    # this cuts the borders off, height and width need to be multiples of 32!
    # image = tf.image.resize(image, [256, 256]) # for local testing
    image = tf.image.crop_to_bounding_box(image, offset_height=44, offset_width=304, target_height=992, target_width=1312)

    return image

# this generator streams the images and masks to the GPU one after another
# it gets a list of pairs that correspond to the directory paths of images and their corresponding masks
# each batch includes a training image, its mask and the 7 augmented versions of it, which are generated on the fly
# batch size means in this case how many original images are loaded for 1 batch,
# the actual batch size is 8 times higher.
# based on https://www.kaggle.com/mukulkr/camvid-segmentation-using-unet
class DataGenerator(Sequence):

    def __init__(self, pair, batch_size, shuffle):
        self.pair = pair
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.map = dict()

    # returns the length of the data set
    def __len__(self):
        return int(tf.math.floor(len(self.pair) / self.batch_size)) * 2

    # returns a single batch
    def __getitem__(self, index):
        # a list that has the indexes of the pairs from which we want to generate images and masks for the batch

        index = index // 2

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [k for k in indexes]

        if index in self.map:
            return self.__data_generation(self.map[index], second_half=True)

        self.map[index] = list_IDs_temp

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    # resets the pair indexes after each epoch and shuffles the indexes so that the batches
    # are in different order for every epoch
    def on_epoch_end(self):
        self.indexes = tf.range(len(self.pair))
        self.map = dict()
        if self.shuffle == True:
            tf.random.shuffle(self.indexes)

    # generates a batch
    def __data_generation(self, list_IDs_temp, second_half=False):
        batch_images = []
        batch_masks = []

        for i in list_IDs_temp:

            # parses the image and the mask of the current pair and then generates the augmented versions
            # it wasn't possible to generate the augmented images beforehand and have completely
            # random images in every batch
            image1 = parse_image(self.pair[i][0])
            batch_images.append(image1)

            mask1 = parse_mask(self.pair[i][1])
            mask1 = to_one_hot(mask1)
            batch_masks.append(mask1)

            image2 = tf.image.flip_left_right(image1)
            batch_images.append(image2)

            mask2 = tf.image.flip_left_right(mask1)
            batch_masks.append(mask2)


            image3 = tf.image.flip_up_down(image1)
            batch_images.append(image3)

            mask3 = tf.image.flip_up_down(mask1)
            batch_masks.append(mask3)


            image4 = tf.image.flip_up_down(image1)
            image4 = tf.image.flip_left_right(image4)
            batch_images.append(image4)

            mask4 = tf.image.flip_up_down(mask1)
            mask4 = tf.image.flip_left_right(mask4)
            batch_masks.append(mask4)

            if second_half:
                # images and masks 1 to 4 but with randomly changed brightness
                delta = tf.random.uniform(shape=[], minval=-0.5, maxval=0.51)
                image5 = tf.image.adjust_brightness(image1, delta)
                batch_images.append(image5)

                batch_masks.append(mask1)


                delta = tf.random.uniform(shape=[], minval=-0.5, maxval=0.51)
                image6 = tf.image.flip_left_right(image1)
                image6 = tf.image.adjust_brightness(image6, delta)
                batch_images.append(image6)

                mask6 = tf.image.flip_left_right(mask1)
                batch_masks.append(mask6)


                delta = tf.random.uniform(shape=[], minval=-0.5, maxval=0.51)
                image7 = tf.image.flip_up_down(image1)
                image7 = tf.image.adjust_brightness(image7, delta)
                batch_images.append(image7)

                mask7 = tf.image.flip_up_down(mask1)
                batch_masks.append(mask7)


                delta = tf.random.uniform(shape=[], minval=-0.5, maxval=0.51)
                image8 = tf.image.flip_up_down(image1)
                image8 = tf.image.flip_left_right(image8)
                image8 = tf.image.adjust_brightness(image8, delta)
                batch_images.append(image8)

                mask8 = tf.image.flip_up_down(mask1)
                mask8 = tf.image.flip_left_right(mask8)
                batch_masks.append(mask8)

                return tf.stack(batch_images[4:]), tf.stack(batch_masks[4:])

        # stack the images and masks of the batch into two tensors
        return tf.stack(batch_images[:4]), tf.stack(batch_masks[:4])

# Data generator that does not augment images, i.e. used for validation and test set
class ValDataGenerator(Sequence):

    def __init__(self, pair, batch_size, shuffle):
        self.pair = pair
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    # returns the length of the data set
    def __len__(self):
        return int(tf.math.floor(len(self.pair) / self.batch_size))

    # returns a single batch
    def __getitem__(self, index):
        # a list that has the indexes of the pairs from which we want to generate images and masks for the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [k for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    # resets the pair indexes after each epoch and shuffles the indexes so that the batches
    # are in different order for every epoch
    def on_epoch_end(self):
        self.indexes = tf.range(len(self.pair))
        if self.shuffle == True:
            tf.random.shuffle(self.indexes)

    # generates a batch
    def __data_generation(self, list_IDs_temp, second_half=False):
        batch_images = []
        batch_masks = []

        for i in list_IDs_temp:

            # parses the image and the mask of the current pair and then generates the augmented versions
            # it wasn't possible to generate the augmented images beforehand and have completely
            # random images in every batch
            image1 = parse_image(self.pair[i][0])
            batch_images.append(image1)

            mask1 = parse_mask(self.pair[i][1])
            mask1 = to_one_hot(mask1)
            batch_masks.append(mask1)

        # stack the images and masks of the batch into two tensors
        return tf.stack(batch_images), tf.stack(batch_masks)

# takes a path to a directory with two sub folders for training images and masks
# and returns a list of pairs of paths for images and the corresponding masks
def make_pairs(path, set):
    pairs = []
    # sorted is very important since os.path.join somehow shuffles the paths and we need
    # the image and mask paths to have the exact same order
    image_paths = sorted(glob(os.path.join(path, set + "_images/*")))
    mask_paths = sorted(glob(os.path.join(path, set + "_masks/*")))
    #image_paths = sorted(glob(os.path.join(path, "test_images2/*")))
    #mask_paths = sorted(glob(os.path.join(path, "test_masks2/*")))

    for i in range(len(image_paths)):
        pairs.append((image_paths[i], mask_paths[i]))

    return pairs



########## LOSS FUNCTION ##########
# based on https://github.com/aruns2120/Semantic-Segmentation-Severstal/blob/master/U-Net/CS2_firstCut.ipynb


# the dice coefficient calculates how much the predicted mask and the correct mask overlap
def dice_coef(y_true, y_predict, smooth=1):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_predict)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_predict):
    return (1 - dice_coef(y_true, y_predict))

# weighted variant of pixelwise_crossentropy
# based on https://www.gitmemory.com/issue/keras-team/keras/6261/569715992
def pixelwise_crossentropy(y_true, y_predicted):#

    # weights that scale the error for each class such that they all have equal impact on the loss
    # important since the data set is very unbalanced
    # weights represent the inverse of the proportion of pixels corresponding to that class in the whole data set
    # needs to be divided by 100.0 to keep the error at a similar magnitude during training
    weight_proton = 132.0 / 100.0
    weight_alpha = 91.0 / 100.0
    weight_V = 311.0 / 100.0
    weight_electron = 71.0 / 100.0
    # weight_proton = 1.0  # for local testing
    # weight_alpha = 1.0
    # weight_V = 1.0
    # weight_electron = 1.0
    weights = [weight_proton, weight_alpha, weight_V, weight_electron]

    # predicted values get scaled such that they are never exactly 0 or 1 since then the logarithm diverges
    y_predicted /= tf.keras.backend.sum(y_predicted, axis=-1, keepdims=True)
    y_predicted = tf.keras.backend.clip(y_predicted,
                                        tf.keras.backend.epsilon(),
                                        1. - tf.keras.backend.epsilon())
    # compute the weighted cross_entropy
    loss = y_true * tf.keras.backend.log(y_predicted)
    loss = -tf.keras.backend.sum(loss * weights, -1)
    return loss

# defines the custom loss function, sum of dice_loss and pixelwise_crossentropy
def pce_dice_loss(y_true, y_predict):
    return pixelwise_crossentropy(y_true, y_predict) + dice_loss(y_true, y_predict)



########## NEURAL NETWORK ##########
# based on https://github.com/aruns2120/Semantic-Segmentation-Severstal/blob/master/U-Net/CS2_firstCut.ipynb


# defines the single convolutional blocks
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

# defines the U-Net architecture
# amount filters controls the amount of filters in the convolutional layers, needs to be a power of 2!
def unet(input, amount_filters):
    # Encoder
    conv_block1 = conv_block(input, amount_filters, 3)
    pooling1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block1)
    # randomly deactivates in each training step 20% of neurons, gives better generalization
    dropout1 = tf.keras.layers.Dropout(0.2)(pooling1)


    conv_block2 = conv_block(dropout1, amount_filters * 2, 3)
    pooling2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block2)
    dropout2 = tf.keras.layers.Dropout(0.2)(pooling2)


    conv_block3 = conv_block(dropout2, amount_filters * 4, 3)
    pooling3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block3)
    dropout3 = tf.keras.layers.Dropout(0.2)(pooling3)


    conv_block4 = conv_block(dropout3, amount_filters * 8, 3)
    pooling4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block4)
    dropout4 = tf.keras.layers.Dropout(0.2)(pooling4)


    encoded_features = conv_block(dropout4, amount_filters * 16, 3)

    # Decoder
    upsample_block1 = tf.keras.layers.UpSampling2D()(encoded_features)
    upsample_block1 = tf.keras.layers.Conv2D(filters=amount_filters * 8, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block1)
    upsample_block1 = tf.keras.layers.concatenate([upsample_block1, conv_block4]) # skip connection
    dropout5 = tf.keras.layers.Dropout(0.2)(upsample_block1)
    conv_block5 = conv_block(dropout5, amount_filters * 8, 3)


    upsample_block2 = tf.keras.layers.UpSampling2D()(conv_block5)
    upsample_block2 = tf.keras.layers.Conv2D(filters=amount_filters * 4, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block2)
    upsample_block2 = tf.keras.layers.concatenate([upsample_block2, conv_block3])
    dropout6 = tf.keras.layers.Dropout(0.2)(upsample_block2)
    conv_block6 = conv_block(dropout6, amount_filters * 4, 3)


    upsample_block3 = tf.keras.layers.UpSampling2D()(conv_block6)
    upsample_block3 = tf.keras.layers.Conv2D(filters=amount_filters * 2, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block3)
    upsample_block3 = tf.keras.layers.concatenate([upsample_block3, conv_block2])
    dropout7 = tf.keras.layers.Dropout(0.2)(upsample_block3)
    conv_block7 = conv_block(dropout7, amount_filters * 2, 3)


    upsample_block4 = tf.keras.layers.UpSampling2D()(conv_block7)
    upsample_block4 = tf.keras.layers.Conv2D(filters=amount_filters, kernel_size=(2, 2),
                                             kernel_initializer="he_normal", padding="same")(upsample_block4)
    upsample_block4 = tf.keras.layers.concatenate([upsample_block4, conv_block1])
    dropout8 = tf.keras.layers.Dropout(0.2)(upsample_block4)
    conv_block8 = conv_block(dropout8, amount_filters, 3)

    # amount of filters in output layer needs to be equal to the amount of classes
    output = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), activation="sigmoid")(conv_block8)
    unet = tf.keras.Model(inputs=[input], outputs=[output])
    return unet



########## TRAINING ##########


# this creates the data generator that is given to the neural net
# batch_size 1 means that 1 original image is used for every batch
# batch_size needs to be a power of 2!
# pairs = make_pairs("/home/mlps_team003/CloudChamberProject")
pairs = make_pairs("C:/Users/lukwi/Desktop/mlps/CloudChamberProject/TrainingSet2", "training")
batch_size = 1
trainset_length = len(pairs) * 2
steps_per_epoch = trainset_length // batch_size

train_generator = DataGenerator(pair=pairs,
                                batch_size=batch_size, shuffle=True)

val_pairs = make_pairs("C:/Users/lukwi/Desktop/mlps/CloudChamberProject/ValidationSet", "validation")
val_generator = ValDataGenerator(pair=val_pairs, batch_size=batch_size, shuffle=True)

test_pairs = make_pairs("C:/Users/lukwi/Desktop/mlps/CloudChamberProject/TestSet", "test")
test_generator = ValDataGenerator(pair=test_pairs, batch_size=batch_size, shuffle=True)

# creates the neural net
# input = tf.keras.layers.Input(shape=[256, 256, 3]) for local testing
input = tf.keras.layers.Input(shape=[992, 1312, 3])  # shape of the input images
unet = unet(input, 8)

# compiles the neural net
optimizer = tf.keras.optimizers.Adam()  # better than stochastic gradient decent
unet.compile(optimizer=optimizer, loss=pce_dice_loss, metrics=[dice_coef, pixelwise_crossentropy])

""" needs to be uncommented to use the weights from the pretrained unsupervised model
# loads the pretrained model and extracts the names of the layers
pretrained_unet = tf.keras.models.load_model("pretraining.h5")
pretrained_layers = pretrained_unet.layers

# copies the pretrained weights to our neural net. We cant copy certain layers since the
# pretraining NN skip connections which changes the amount of parameters of some layers
for layer in pretrained_layers:
    if (layer.name != "conv2d_11" and layer.name != "conv2d_14"
            and layer.name != "conv2d_17" and layer.name != "conv2d_20"
            and layer.name != "conv2d_22"):
        untrained_layer = unet.get_layer(name=layer.name)  # retrieves the untrained layer
        pretrained_layer = pretrained_unet.get_layer(name=layer.name)  # retrieves the trained layer
        untrained_layer.set_weights(pretrained_layer.get_weights())  # copies weights
"""

# callback for logging training metrics that can be displayed in Tensorboard
# with the command 'Tensorboard --logdir training_logs/train'
metrics_logger = tf.keras.callbacks.TensorBoard(log_dir="./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"), update_freq='epoch', write_images=True)

# callback for saving the best model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoints/pretraining', monitor='val_loss', save_best_only=True)

# trains it
unet_history = unet.fit(
    x=train_generator,
    epochs=40,
    steps_per_epoch=steps_per_epoch,
    callbacks=[metrics_logger, model_checkpoint],
    validation_data=val_generator
)

# Evaluates it
results = unet.evaluate(test_generator)
print("Test loss, dice_coef and pixelwise crossentropy: ", results)

# saves the computed weights and the network architecture
# when loading the neural net from the h5 file it first needs to be recompiled
# since tensorflow has trouble with the custom loss function
unet.save("unet.h5")