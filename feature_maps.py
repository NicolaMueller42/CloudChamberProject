import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# This script was used for visualizing the feature maps of the
# neural networks. It works like the predictions script but the
# difference is that we change the output layer of the loaded
# model to be an intermediate layer.


def parse_image(name):
    image = tf.io.read_file(name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.crop_to_bounding_box(image, offset_height=44, offset_width=304, target_height=992, target_width=1312)
    return image

#define the custom loss function

def pixelwise_crossentropy(y_true, y_predicted):
    y_predicted /= tf.keras.backend.sum(y_predicted, axis=-1, keepdims=True)
    y_predicted = tf.keras.backend.clip(y_predicted,
                                        tf.keras.backend.epsilon(),
                                        1-tf.keras.backend.epsilon())
    loss = y_true * tf.keras.backend.log(y_predicted)
    loss = -tf.keras.backend.sum(loss, -1)
    return loss

def dice_coef(y_true, y_predict, smooth=1):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_predict)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_predict):
    return (1 - dice_coef(y_true, y_predict))

def pce_dice_loss(y_true, y_predict):
    return pixelwise_crossentropy(y_true, y_predict) + dice_loss(y_true, y_predict)

unet = tf.keras.models.load_model("transfer_learning.h5", compile=False)

unet.compile(optimizer="adam", loss=pce_dice_loss, metrics=[dice_coef])

unet.summary()

# specify the intermediate layer from which we want to see the feature map
FeatureMapLayer = 'conv2d_22'

# define the new output layer
unet = tf.keras.models.Model(inputs=unet.input, outputs=unet.get_layer(name=FeatureMapLayer).output)

# specify the paths to the images
path = "C:/Users/Nicola/Desktop/Uni/MLproject/"
test_images_paths = sorted(glob(os.path.join(path, "FeatureMapExamples/*")))

# parse the images
test_images = []
for x in test_images_paths:
    test_images.append(parse_image(x))

# starts the time counter
start = time.process_time()

fig = plt.figure()
fig.suptitle(FeatureMapLayer)
i = 0
for image in test_images:
    #plt.imshow(image)
    #plt.show()

    # stack the image (adds a dimension with value 1 for the batch size) and get the prediction
    prediction = unet.predict(tf.stack([image]))

    # dimensions of the generated plot
    height = 2
    length = 2
    ix = 1
    for _ in range(height):
        for _ in range(length):
            # we create a subplot for each feature map
            ax = plt.subplot(height, length, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(prediction[0, :, :, ix-1], cmap='gray')
            ix += 1

    plt.show()

    # print passed time in seconds
    print((time.process_time() - start) / 10.0)
