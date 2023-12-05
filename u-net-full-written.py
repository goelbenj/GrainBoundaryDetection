#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
data_dir = Path('./GRAIN_DATA_SET')
# for dirname, _, filenames in os.walk(data_dir):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
from tqdm import tqdm 


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
DATA_SIZE = 480
NUM_EPOCHS = 30
TRAIN = False
# set seed
random.seed(2023)

# Set the directories containing the images and masks
image_dir = os.path.join(data_dir, 'RG')
mask_dir = os.path.join(data_dir, 'RGMask')

# Set the target image size
target_size = (IMG_WIDTH, IMG_HEIGHT)

# Create empty lists to hold the images and masks
images = []
masks = []

# Iterate through the directories and load the images and masks
for i, file in enumerate(sorted(os.listdir(image_dir))):
    if i == DATA_SIZE:
        break
    # Load the image and resize to the target size
    img = np.array(PIL.Image.open(os.path.join(image_dir, file)))
    img = tf.image.resize(img, target_size).numpy()
    
    # Append the resized image to the list of images
    images.append(img)
    
for i, file in enumerate(sorted(os.listdir(mask_dir))):
    if i == DATA_SIZE:
        break
    # Load the corresponding mask and resize to the target size
    #mask_file = file.replace('.jpg', '.png')
    mask = np.array(PIL.Image.open(os.path.join(mask_dir, file)))
    mask = np.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, target_size).numpy()
    
    # Append the resized mask to the list of masks
    masks.append(mask)
    

image_x = random.randint(0, DATA_SIZE)
image_x

imshow(images[image_x])
print((images[image_x] / 255))
# plt.show()
imshow(masks[image_x])
# plt.show()

# create the X and Y (input and output)

X_train = np.array(images)
Y_train = np.array(masks)

# change the Y to a boolean
Y_train = np.where(Y_train > 245, True, False)

mask_length = len(masks)
#convert the boolean where it is true (any of the 3 channels) to a (336, 128, 128, 1)
#basically reduce the 3 channel dimension RGB to just one boolean value

Y_t= np.any(Y_train, axis=-1)
Y_t = Y_t.reshape(mask_length, IMG_WIDTH, IMG_HEIGHT, 1)
################################

if TRAIN:
    #Build the model

    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # create the checkpoint path

    checkpoint_path = 'checkpoint_path/GrainsTraining.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #Modelcheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]

    results = model.fit(X_train, Y_t, validation_split=0.1, batch_size=16, epochs=NUM_EPOCHS, callbacks=callbacks)
    model.save('Grains_DETECTION_UNET.h5')
else:
    model = tf.keras.models.load_model("Grains_DETECTION_UNET.h5")

####################################

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
#preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
#preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
# plt.show()
imshow(np.squeeze(Y_t[ix]))
# plt.show()
imshow(np.squeeze(preds_train_t[ix]))
# plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
# plt.show()
imshow(np.squeeze(Y_t[int(Y_train.shape[0]*0.9):][ix]))
# plt.show()
imshow(np.squeeze(preds_val_t[ix]))
# plt.show()

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

def show_predictions(dataset=None, num=1, save_name=None):
    if dataset is not None:
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        for i, sample in enumerate(range(-num, 0)):
            file_name = os.path.join(save_name, f'{i}.jpg')
            images, masks = X_train[sample], Y_t[sample]
            pred_masks = preds_val_t[sample]
            display([images, masks, pred_masks], file_name)

# Predictions
show_predictions(preds_val_t, 10, save_name='./u-net-outputs/infer')

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
    TOTAL_IMAGE_AREAS = len(masks) * IMG_WIDTH * IMG_HEIGHT
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


Y_t = Y_t.astype(np.uint8)
print("Baseline Statistics:")
calc_grain_dimension_measurements(Y_t)
print("Test Model Statistics:")
calc_grain_dimension_measurements(preds_train_t)
