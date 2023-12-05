#!/usr/bin/env python
# coding: utf-8

# Installing Dependencies

# ! pip install transformers
# ! pip install tensorflow


# Imports
import cv2
import os
import tensorflow as tf
from tensorflow.keras import backend
import math
import matplotlib.pyplot as plt
import numpy as np

# set random seed
tf.random.set_seed(2023)

# config
batch_size = 4
image_size = 128
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])
# lr = 0.00006
lr = 0.001
epochs = 1
train = False

# Dataset
# Function to read the image file
def load_image_file(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_jpeg(image, channels=3)
    mask = tf.image.decode_png(mask, channels=1)

    return {"image": image, "segmentation_mask": mask}


# Loading the dataset
train_image_dir = "./GRAIN_DATA_SET/RG"
train_mask_dir = "./GRAIN_DATA_SET/RGMask"
valid_image_dir = train_image_dir
valid_mask_dir = train_mask_dir
test_image_dir = train_image_dir
test_mask_dir = train_mask_dir

# Define list of image and mask file names
train_image_names = sorted(os.listdir(train_image_dir))
train_mask_names = sorted(os.listdir(train_mask_dir))

valid_image_names = sorted(os.listdir(valid_image_dir))
valid_mask_names = sorted(os.listdir(valid_mask_dir))

test_image_names = sorted(os.listdir(test_image_dir))
test_mask_names = sorted(os.listdir(test_mask_dir))

train_pairs = []
for img_name in train_image_names:
    # Check if image file name matches mask file name
    mask_name = img_name.replace("RG", "RGMask")
    if mask_name in train_mask_names:
        train_pairs.append((os.path.join(train_image_dir, img_name), os.path.join(train_mask_dir, mask_name)))
        
valid_pairs = []
for img_name in valid_image_names:
    # Check if image file name matches mask file name
    mask_name = img_name.replace("RG", "RGMask")
    if mask_name in valid_mask_names:
        valid_pairs.append((os.path.join(valid_image_dir, img_name), os.path.join(valid_mask_dir, mask_name)))

test_pairs = []
for img_name in test_image_names:
    # Check if image file name matches mask file name
    mask_name = img_name.replace("RG", "RGMask")
    if mask_name in test_mask_names:
        test_pairs.append((os.path.join(test_image_dir, img_name), os.path.join(test_mask_dir, mask_name)))

# Load image and mask data from file paths
data_train = [load_image_file(image_path, mask_path) for image_path, mask_path in train_pairs]
data_valid = [load_image_file(image_path, mask_path) for image_path, mask_path in valid_pairs]
data_test = [load_image_file(image_path, mask_path) for image_path, mask_path in test_pairs]

len(data_train), len(data_valid), len(data_test)


# Normalization and Image Resizing
# P.S. You could do data augmentation here as well. I kept it very simple
def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    input_mask = tf.where(input_mask > 245, True, False)
    input_mask = tf.math.reduce_any(input_mask, axis=-1)
    input_mask = tf.cast(input_mask, dtype=tf.uint8)
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )
    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))
    
    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}

train_data = [load_image(datapoint) for datapoint in data_train]
valid_data = [load_image(datapoint) for datapoint in data_valid]
test_data = [load_image(datapoint) for datapoint in data_test]

# Visualize sample (uncomment for debug)
# index = 120
# plt.figure()
# plt.imshow(train_data[index]["labels"])
# plt.show()
# plt.figure()
# plt.imshow(tf.keras.utils.array_to_img(tf.transpose(train_data[index]['pixel_values'], (1, 2, 0))))
# plt.show()

def generator_train():
    for datapoint in train_data:
        yield datapoint

def generator_valid():
    for datapoint in valid_data:
        yield datapoint

def generator_test():
    for datapoint in test_data:
        yield datapoint


# Using <code>tf.data.Dataset</code> to build input pipeline
auto = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_generator(generator_train, output_types={"pixel_values": tf.float32, "labels": tf.int32}).cache().shuffle(batch_size * 10).batch(batch_size).prefetch(auto)

valid_ds = tf.data.Dataset.from_generator(generator_valid, output_types={"pixel_values": tf.float32, "labels": tf.int32}).batch(batch_size).prefetch(auto)

test_ds = tf.data.Dataset.from_generator(generator_test, output_types={"pixel_values": tf.float32, "labels": tf.int32}).batch(batch_size).prefetch(auto)
print(train_ds.element_spec)


# Visualizing the data
def display(display_list, save_name=None):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    if save_name is not None:
        plt.savefig(save_name)
    # plt.show()

# Uncomment for debugging
# for samples in train_ds.take(2):
#     sample_image, sample_mask = samples["pixel_values"][0], samples["labels"][0]
#     sample_image = tf.transpose(sample_image, (1, 2, 0))
#     sample_mask = tf.expand_dims(sample_mask, -1)
#     display([sample_image, sample_mask])
#     print(sample_image.shape)


# Model
from transformers import TFSegformerForSemanticSegmentation

model_checkpoint = './pretrained/mit-b0'
id2label =  {0: "outer", 1: "landslide"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)
model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Callback to visualize image after every epoch
from IPython.display import clear_output


def create_mask(pred_mask):
    pred_mask = tf.transpose(pred_mask, (0, 2, 3, 1))
    # Must resize output mask to original image dimensions
    pred_mask = tf.image.resize(pred_mask, (image_size, image_size), method=tf.image.ResizeMethod.BILINEAR)
    pred_mask = tf.transpose(pred_mask, (0, 3, 1, 2))
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    pred_mask = tf.cast(pred_mask, dtype=tf.uint8)
    return pred_mask

def model_predict(model, sample):
    images, masks = sample["pixel_values"], sample["labels"]
    masks = tf.expand_dims(masks, -1)
    pred_masks = model.predict(images, verbose=1).logits
    images = tf.transpose(images, (0, 2, 3, 1))
    return images, masks, pred_masks

def show_predictions(dataset=None, num=1, save_name=None):
    if dataset:
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        # for i, sample in enumerate(dataset.take(num)):
        for i, sample in enumerate(range(-num, 0)):
            sample = dataset[sample]
            temp_sample = dict()
            temp_sample['labels'] = tf.expand_dims(sample['labels'], axis=0)
            temp_sample['pixel_values'] = tf.expand_dims(sample['pixel_values'], axis=0)
            file_name = os.path.join(save_name, f'{i}.jpg')
            images, masks, pred_masks = model_predict(model, temp_sample)
            display([images[0], masks[0], create_mask(pred_masks[0:1])[0]], file_name)
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(tf.expand_dims(sample_image, 0))),
            ],
            save_name
        )


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset, save_name=f'./outputs/train/epoch{epoch+1}')
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


# Training Loop
export_path = "./weights/segformer-5-b0.h5"
if train:
    # Hyperparameters and compiling the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        callbacks=[DisplayCallback(test_data)],
        epochs=epochs,
    )
    model.save_weights(export_path)

    # Loss Plot
    print(plt.style.available)
    plt.style.use("seaborn-v0_8")

    def display_training_curves(training, validation, title, subplot):
        ax = plt.subplot(subplot)
        ax.plot(training)
        ax.plot(validation)
        ax.set_title('Model '+ title)
        ax.set_ylabel(title)
        ax.set_xlabel('epoch')
        ax.legend(['training', 'validation'])

    plt.subplots(figsize=(8,8))
    plt.tight_layout()
    display_training_curves(history.history['loss'], history.history['val_loss'], 'Loss', 111)
    plt.savefig("train_eval_plot_segformer-5-b1.jpg")
else:
    model.load_weights(export_path)

# Predictions
show_predictions(valid_data, 10, save_name='./outputs/infer')

def eval(model, dataset):
    result_masks = tf.reshape(tf.constant([], dtype=tf.uint8),
                              (0, image_size, image_size, 1))
    for i, sample in enumerate(dataset):
        images, masks, pred_masks = model_predict(model, sample)
        pred_masks = create_mask(pred_masks)
        result_masks = tf.concat([result_masks, pred_masks], axis=0)
    return result_masks

Y_t = tf.convert_to_tensor([x['labels'] for x in generator_valid()])
Y_t = tf.expand_dims(Y_t, axis=-1).numpy()
preds_train_t = eval(model, valid_ds).numpy()
# preds_val_t = eval(model, valid_ds).numpy()

def calc_dice_score(real_mask, pred_mask):
    # calculcate dice coefficients
    # Initialize a list to store the dice coefficients for each mask
    dice_coefficients = []

    # Iterate through the masks in both directories
    for i in range(len(pred_mask)):
        # Calculate the intersection of the masks
        intersection = np.sum(pred_mask[i] * real_mask[i])

        # Calculate the size of each mask
        predicted_mask_size = np.sum(pred_mask[i])
        real_mask_size = np.sum(real_mask[i])

        # Calculate the dice coefficient for the two masks
        dice = 2 * intersection / (predicted_mask_size + real_mask_size)

        # Add the dice coefficient to the list
        dice_coefficients.append(dice)

        # Calculate the average dice coefficient for the set of masks
        average_dice_coefficient = np.mean(dice_coefficients)

    print(f'Average dice coefficient for the data it was trained on: {average_dice_coefficient:.4f}')

calc_dice_score(Y_t, Y_t)
calc_dice_score(Y_t, preds_train_t)

# Now the measurement for LOOPS, one for each of the variables 3 total
# 1. circularity
# 2. max/min
# 3. X/Y (aspect ratio)

# This loop reads all the images and then stores the area of each grain for the image in an array
# Circularity, max diameter, min diameter, and aspect ratio for all masks
def calc_grain_dimension_measurements(masks):
    CR = []
    MAX = []
    MIN = []
    AR = []
    X = []
    Y = []
    AREAS = []
    NUM_GRAINS = 0
    TOTAL_IMAGE_AREAS = len(masks) * image_size * image_size
    TOTAL_GRAIN_AREA_PERCENT = 0
    TOTAL_GRAIN_BOUNDARY_AREA_PERCENT = 1 - TOTAL_GRAIN_AREA_PERCENT
    for pred in masks:
        # pred *= 255
        # ret, thresh = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # count total number of grains across all masks
        NUM_GRAINS += len(contours)

        # Iterate through all the contours
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            AREAS.append(area)
            perimeter = cv2.arcLength(contour, True)
            # Only calculate metrics for non-pointwise contours
            if not perimeter:
                continue
            circularity_ratio = (4 * math.pi * area) / (perimeter ** 2)
            CR.append(circularity_ratio)

            # Calculate the convex hull of the contour
            hull = cv2.convexHull(contour)
            # Initialize the maximum and minimum distances to zero
            max_distance = 0
            min_distance = float('inf')

            # Iterate through all pairs of points on the convex hull
            for i in range(len(hull)):
                for j in range(i+1, len(hull)):
                    # Calculate the distance between the two points
                    distance = np.sqrt((hull[i][0][0] - hull[j][0][0])**2 + (hull[i][0][1] - hull[j][0][1])**2)

                    # Update the maximum and minimum distances if necessary
                    if distance > max_distance:
                        max_distance = distance
                    if distance < min_distance:
                        min_distance = distance

            # Add the maximum and minimum distances for the current contour to the list of distances
            MAX.append([max_distance])
            MIN.append([min_distance])

            # Get the bounding rectangle of the grain
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the x-diameter of the grain using the width of the bounding rectangle
            X.append([w])
            Y.append([h])
            aspect_ratio = w/h
            AR.append([aspect_ratio])
    # calculate total grain area percent
    TOTAL_GRAIN_AREA_PERCENT = np.sum(AREAS) / TOTAL_IMAGE_AREAS
    # calculate total grain boundary area percent
    TOTAL_GRAIN_BOUNDARY_AREA_PERCENT = 1 - TOTAL_GRAIN_AREA_PERCENT

    # Print results
    print(f"Average Circularity: {np.mean(CR)}")
    print(f"Circularity Std. Dev: {np.std(CR)}")
    print(f"Average Max Diameter {np.mean(MAX)}")
    print(f"Max Diameter Std. Dev: {np.std(MAX)}")
    print(f"Average Min Diameter {np.mean(MIN)}")
    print(f"Min Diameter Std. Dev: {np.std(MIN)}")
    print(f"Average Aspect Ratio {np.mean(AR)}")
    print(f"Aspect Ratio Std. Dev: {np.std(AR)}")
    print(f"Average X-Diameter {np.mean(X)}")
    print(f"X-Diameter Std. Dev: {np.std(X)}")
    print(f"Average Y-Diameter {np.mean(Y)}")
    print(f"Y-Diameter Std. Dev: {np.std(Y)}")
    print(f"Average Aspect Ratio {np.mean(AR)}")
    print(f"Aspect Ratio Std. Dev: {np.std(AR)}")
    print(f"Total Grains: {NUM_GRAINS}")
    print(f"Average Individual Grain Area: {np.mean(AREAS)}")
    print(f"Individual Grain Area Std. Dev: {np.std(AREAS)}")
    print(f"Total Grain Area Percent: {TOTAL_GRAIN_AREA_PERCENT}")
    print(f"Total Grain Boundary Area Percent: {TOTAL_GRAIN_BOUNDARY_AREA_PERCENT}")


print("Baseline Statistics:")
calc_grain_dimension_measurements(Y_t)
print("Test Model Statistics:")
calc_grain_dimension_measurements(preds_train_t)
