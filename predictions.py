import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import os
import time
# turn off GPU processing because
# tensorflow-gpu can lead to trouble if not installed correctly
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# This script loads a trained neural network and then displays the predictions for given images


# turns the predicted class probability maps into RGB masks
def create_mask(one_hot):
    # input is a tensor that contains the class probability maps

    # colors for the predicted mask, need to be divided by 255.0 since we work with float colors
    colors = [
        [227.0 / 255.0, 26.0 / 255.0, 28.0 / 255.0],  # Red for Protons
        [0.0 / 255.0, 100.0 / 255.0, 0.0 / 255.0],  # Green for Alpha particles
        [139.0 / 255.0, 0.0 / 255.0, 139.0 / 255.0],  # Violet for Alpha_V particles
        [0.0 / 255.0, 0.0 / 255.0, 205.0 / 255.0],  # Blue for electrons
        [0.0, 0.0, 0.0]  # Black
    ]

    # unstack to get a list of class probability maps, where each cell has the predicted probability
    # that the corresponding pixel has the color of the class
    raw_channels = tf.unstack(one_hot, axis=-1)

    # apply thresholds, every probability lower than 50% is set to 0 since we only want confident predictions
    # changing the thresholds changes the final masks a lot
    threshold_channel0 = tf.math.maximum(raw_channels[0] - 0.8, 0)  # Proton
    threshold_channel1 = tf.math.maximum(raw_channels[1] - 0.5, 0)  # Alpha
    threshold_channel2 = tf.math.maximum(raw_channels[2] - 0.5, 0)  # V
    threshold_channel3 = tf.math.maximum(raw_channels[3] - 0.5, 0)  # Electron

    # build a map with 0s everywhere so that we have a class map for the black pixels
    # since the neural network does not predict a class map for black
    # first adds all channels together so we know which pixels are everywhere predicted as 0
    black_map = tf.math.add(threshold_channel0, threshold_channel1)
    black_map = tf.math.add(black_map, threshold_channel2)
    black_map = tf.math.add(black_map, threshold_channel3)
    # creates a map with 1s where the pixels had color probability 0
    black_map = tf.equal(black_map, 0.0)
    # cast to float 32
    black_map = tf.cast(black_map, tf.float32)

    # stack the class maps with the thresholds applied to a tensor so that
    # we can later apply the arg max function
    threshold_one_hot = tf.stack([threshold_channel0, threshold_channel1,
                                  threshold_channel2, threshold_channel3,
                                  black_map], axis=-1)

    # black, red, green, violet, blue
    class_indices = [0, 1, 2, 3, 4]
    # returns a 2d class map where in every pixel stands the class index of the
    # class that had the highest probability for that pixel
    argmax_mask = tf.argmax(threshold_one_hot, axis=-1)

    # now for every class index we create a 0-1 encoded (one hot) class map
    # so that when we later multiply by the RGB color we get for each pixel either black
    # or the correct color
    channels = []
    for index in class_indices:
        # returns one hot class map where a pixel has value 1 if the corresponding
        # class was the one with highest probability for that pixel
        class_map = tf.equal(argmax_mask, index)
        class_map = tf.cast(class_map, tf.float32)
        channels.append(class_map)

    # stack 3 copies of a one hot class map together and multiply by the corresponding
    # RGB color to get an RGB class map
    red = tf.stack([channels[0], channels[0], channels[0]], axis=-1) * colors[0]
    green = tf.stack([channels[1], channels[1], channels[1]], axis=-1) * colors[1]
    violet = tf.stack([channels[2], channels[2], channels[2]], axis=-1) * colors[2]
    blue = tf.stack([channels[3], channels[3], channels[3]], axis=-1) * colors[3]
    black = tf.stack([channels[4], channels[4], channels[4]], axis=-1) * colors[4]

    # add all the RGB class maps together to get the final mask
    mask = tf.math.add(red, green)
    mask = tf.math.add(mask, violet)
    mask = tf.math.add(mask, blue)
    mask = tf.math.add(mask, black)

    return mask

# parses the image and crops the borders
def parse_image(name):
    image = tf.io.read_file(name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize(image, [256, 256])
    image = tf.image.crop_to_bounding_box(image, offset_height=44, offset_width=304, target_height=992, target_width=1312)
    return image

# we need to define the PCE dice loss function again since we need to recompile the loaded neural network
def dice_coef(y_true, y_predict, smooth=1):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_predict)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_predict):
    return (1 - dice_coef(y_true, y_predict))

def pixelwise_crossentropy(y_true, y_predicted):#
    weight_proton = 188.0 / 100.0
    weight_alpha = 97.0 / 100.0
    weight_V = 219.0 / 100.0
    weight_electron = 77.0 / 100.0
    weights = [weight_proton, weight_alpha, weight_V, weight_electron]

    y_predicted /= tf.keras.backend.sum(y_predicted, axis=-1, keepdims=True)
    y_predicted = tf.keras.backend.clip(y_predicted,
                                        tf.keras.backend.epsilon(),
                                        1-tf.keras.backend.epsilon())
    loss = y_true * tf.keras.backend.log(y_predicted)
    loss = -tf.keras.backend.sum(loss * weights, -1)
    return loss

def pce_dice_loss(y_true, y_predict):
    return pixelwise_crossentropy(y_true, y_predict) + dice_loss(y_true, y_predict)


# load the model from the h5 file, compile needs to be set to false
model_name = "training.h5"
unet = tf.keras.models.load_model(model_name, compile=False)

# get a summary of the model layers and parameters
#unet.summary()

# compile manually
unet.compile(optimizer="adam", loss=pce_dice_loss, metrics=[dice_coef])

# specify the paths to the images
path = "C:/Users/lukwi/Desktop/mlps/CloudChamberProject/TestSet"
test_images_paths = sorted(glob(os.path.join(path, "test_images/*")))


# parse the images
test_images = []
for x in test_images_paths:
    test_images.append(parse_image(x))

# starts the time counter
start = time.process_time()

plt.figure()
i = 0
for image in test_images:
    plt.imshow(image)
    plt.show()

    # stack the image (adds a dimension with value 1 for the batch size) and get the prediction
    prediction = unet.predict(tf.stack([image]))

    # For visualizing pretraining
    # prediction = tf.reshape(prediction, [992, 1312, 3])
    # plt.imshow(prediction)
    # plt.show()

    # reshape the prediction to drop the batch size dimension and then turn it into a RGB mask
    predicted_mask = create_mask(tf.reshape(prediction, [992, 1312, 4]))

    plt.title(model_name + " prediction")
    plt.imshow(predicted_mask)
    plt.show()

    # add image and prediction together, multiply original image by 0.9 to make it a bit darker
    # so that the colors of the mask are more visible
    plt.title(model_name + " final output")
    plt.imshow(tf.math.add(predicted_mask, image * 0.9))
    # plt.savefig('./videos/normal/' + str(i) + '.jpg')
    i += 1
    plt.show()

    # print passed time in seconds
    print((time.process_time() - start) / 10.0)
