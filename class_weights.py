import tensorflow as tf
from glob import glob
import os
# turn off GPU processing because
# tensorflow-gpu can lead to trouble if not installed correctly
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# This script computes the loss weights for every class.
# These weights balance the loss and thus make the neural network focus
# on rare and common particles equally much.


# parses the masks and crops the borders
def parse_mask(name):
    mask = tf.io.read_file(name)
    mask = tf.image.decode_png(mask)
    mask = tf.image.crop_to_bounding_box(mask, offset_height=44, offset_width=304, target_height=992, target_width=1312)
    return mask

# gets the paths of the masks in a given directory
def get_mask_paths(path):
    mask_paths = sorted(glob(os.path.join(path, "training_masks/*")))
    return mask_paths

mask_paths = get_mask_paths("C:/Users/lukwi/Desktop/mlps/CloudChamberProject/TrainingSet2")

# the RGB values of the colors in the masks
colors = [
    [227, 26, 28],  # Red
    [65, 117, 5],  # Green
    [106, 61, 154],  # Violet
    [31, 120, 180]  # Blue
]
# initializes the counter for the pixels of the different colors
color_pixels = [0, 0, 0, 0]

total_pixels = 992 * 1312 * len(mask_paths)

# parses each mask individually and computes the number of pixels of every color
mask_counter = 0
for mask_path in mask_paths:
    mask = parse_mask(mask_path)
    mask_counter = mask_counter + 1
    print(mask_counter)
    # looks at each color individually
    for color_index in range(len(colors)):
        # tf.equal compares every pixel of the mask with the current RGB color
        # and returns a matrix where a cell is TRUE if the corresponding pixel had the color of the current class
        # reduce all then turns this boolean matrix into a 2D map
        color_mask = tf.reduce_all(tf.equal(mask, colors[color_index]), axis=-1)
        # casts the boolean color map to integers and the sums all cells with value 1 up
        color_pixel_count = tf.reduce_sum(tf.cast(color_mask, tf.int64))
        # updates the pixel counter for the current class
        color_pixels[color_index] = color_pixels[color_index] + color_pixel_count

# computes the class weights as the inverse of the proportion that each color had among all pixels
red_weight = (1 / color_pixels[0]) * total_pixels
green_weight = (1 / color_pixels[1]) * total_pixels
violet_weight = (1 / color_pixels[2]) * total_pixels
blue_weight = (1 / color_pixels[3]) * total_pixels

print(str(mask_counter) + " masks analyzed.")
print("weight for protons: " + str(red_weight))
print("weight for alphas: " + str(green_weight))
print("weight for Vs: " + str(violet_weight))
print("weight for electrons: " + str(blue_weight))
