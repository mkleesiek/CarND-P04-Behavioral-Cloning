import csv
import math
import glob
import os
import os.path
import argparse

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Cropping2D, Lambda, BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def update_progress(progress, opt_text=""):
    """
    Print a rewritable single line ASCI progress bar.
    :param progress: Progress from 0.0 to 1.0.
    """
    bar_length = 20
    block = int(math.ceil(bar_length * progress))
    print("Progress: [{}] {:.1f}%   {:s}".format(
        "#" * block + "-" * (bar_length - block), progress * 100, opt_text), end='\r')


def parse_driving_logs(data_dir, flip=True, cameras=[0,1,2], steering_correction=0.2):
    """
    Parse all CSV driving logs and corresponding image data recorded with the training simulator.

    :param data_dir: Folder containing the training data.
    :param flip: If True, offline augmentation is performed and a horizontally flipped version of each image
    is added to the result list.
    :param cameras: List containing the CSV column indices corresponding to camera angles (center, left, right).
    :param steering_correction: Steering angle correction, which is added to left (subtracted from right) camera images.
    :return: A list of tuples (absolute filename, steering angle)
    """

    IMG_DIR = data_dir + '/IMG'
    IMG_FLIP_DIR = data_dir + '/IMG_flip'

    if not os.path.exists(data_dir):
        return

    # create directory for offline augmented data if necessary
    if flip and not os.path.exists(IMG_FLIP_DIR):
        os.mkdir(IMG_FLIP_DIR)

    samples = []

    # iterate over all CSV files found in the data directory
    for csv_path in glob.glob(data_dir + '/*.csv'):

        print("Parsing {}.".format(csv_path))

        with open(csv_path) as csv_file:

            num_lines = sum(1 for line in csv_file)
            csv_file.seek(0)

            reader = csv.reader(csv_file)
            # skip header row
            next(reader)

            # iterate over all lines
            for r, line in enumerate(reader):

                if r % 100 == 0 or num_lines - r < 10:
                    update_progress(r/num_lines)

                # iterate over all referenced images (columns 0, 1 and 2)
                for i in cameras:
                    basename = line[i].split('/')[-1]
                    img_path = IMG_DIR + '/' + basename

                    if not os.path.isfile(img_path):
                        continue

                    angle = float(line[3])
                    # apply steering correction angle to left and right camera pictures
                    if i == 1:
                        angle = angle + steering_correction
                    elif i == 2:
                        angle = angle - steering_correction

                    samples.append((img_path, angle))

                    # generate a horizontally flipped variant of the image and save it to
                    # filesystem, if it doesn't exist yet
                    if flip:
                        img_flip_path = IMG_FLIP_DIR + '/' + basename
                        if not os.path.isfile(img_flip_path):
                            img_rgb = Image.open(img_path)
                            img_rgb_flip = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                            img_rgb_flip.save(img_flip_path)

                        samples.append((img_flip_path, -angle))

    return samples


def create_model(input_shape=(160, 320, 3), vertical_cropping=(74, 20), dropout_rate=0.2):
    """
    Creates a `Sequential` model instance based on the CNN architecture presented in NVIDIA's
    paper 'End to End Learning for Self-Driving Cars' (https://arxiv.org/pdf/1604.07316v1.pdf).

    :param input_shape: Shape of the input image data (width, height, colors).
    :param vertical_cropping: Number of pixels to be cropped from (top, bottom).
    :param dropout_rate: Dropout rate used for dropout layers between fully connected layers
    of the CNN (only used during training).
    :return: A `keras.models.Sequential` instance.
    """

    model = Sequential()

    # apply top/bottom crop
    model.add(Cropping2D(cropping=(vertical_cropping, (0, 0)), input_shape=input_shape))

    # series of convolutional layers with strides, normalization and RELU activation
    model.add(BatchNormalization())
    model.add(Conv2D(24, 5, strides=(2, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, 5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, 5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))

    # flatten input nodes
    model.add(Flatten())

    # series of fully connected layers including dropouts
    model.add(Dropout(dropout_rate))
    model.add(Dense(100))

    model.add(Dropout(dropout_rate))
    model.add(Dense(50))

    model.add(Dropout(dropout_rate))
    model.add(Dense(10))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    return model


def plot_history(history_object, to_file=None):
    """
    Plot the training and testing loss for each epoch in the provided history.
    """

    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    if to_file:
        plt.savefig(to_file)
    else:
        plt.show()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create driving model from training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_folder',
        type=str,
        help='Path to folder containing training logs and images.'
    )
    parser.add_argument(
        '--steering-correction', dest='steering_correction',
        type=float,
        default=0.2,
        help='Steering correction angle added/subtracted from left/right camera angles.'
    )
    parser.add_argument(
        '--epochs', dest='epochs',
        type=int,
        default=10,
        help='Number of epochs used for training the model.'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size',
        type=int,
        default=64,
        help='Batch size of samples used for training.'
    )
    args = parser.parse_args()

    print("Command line arguments:")
    print(vars(args))

    # parse driving logs and perform offline augmentation
    samples = parse_driving_logs(args.data_folder, steering_correction=args.steering_correction)

    # convert list of images and steering angles into a pandas dataframe and split into training and validation set
    df = pd.DataFrame(samples, columns=['filename', 'angle'])
    df_train, df_valid = train_test_split(df, shuffle=True, test_size=0.2)

    # prepare data generator for training data, including in-place online augmentation (rotation, shear and zoom)
    datagen_train = ImageDataGenerator(
        rotation_range=5.0,
        shear_range=5.0,
        zoom_range=0.05,
        fill_mode='nearest')
    train_generator = datagen_train.flow_from_dataframe(dataframe=df_train, x_col='filename', y_col='angle',
        class_mode='raw', target_size=(160, 320), interpolation='bilinear', batch_size=args.batch_size)

    # prepare data generator for validation data, without augmentation
    datagen_valid = ImageDataGenerator()
    valid_generator = datagen_valid.flow_from_dataframe(dataframe=df_valid, x_col='filename', y_col='angle',
        class_mode='raw', target_size=(160, 320), interpolation='bilinear', batch_size=args.batch_size)

    # instantiate the CNN model and print a summary
    model = create_model()
    model.summary()
    plot_model(model, to_file=args.data_folder + '/model.png',
        show_shapes=True, show_layer_names=True, rankdir="TB", expand_nested=False, dpi=96)

    # compile the model using a mean-squared error loss function and an adaptive Adam optimizer
    model.compile(loss='mse', optimizer='adam')

    # calculate the number of in-place augmented samples per batch
    # double the amount of augmented training images (using each raw training image twice)
    step_size_train = 2 * train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size

    # perform training and validation
    history = model.fit(
        train_generator,
        steps_per_epoch=step_size_train,
        validation_data=valid_generator,
        validation_steps=step_size_valid,
        workers=os.cpu_count(),
        shuffle=True,
        epochs=args.epochs,
        verbose=1)

    # save model in HDF5 format
    model.save(args.data_folder + '/model.h5')
    print("Model saved.")

    # plot loss vs. epochs
    plot_history(history, args.data_folder + '/loss.png')


if __name__ == "__main__":
    main()
