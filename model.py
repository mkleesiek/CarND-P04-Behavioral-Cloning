import csv
import math
import glob
import os
import os.path

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D

import matplotlib.pyplot as plt

data_dir = 'data_custom'
img_dir = data_dir + '/IMG'
img_flip_dir = data_dir + '/IMG_flip'

STEERING_CORRECTION = 0.2

NB_EPOCHS = 10
BATCH_SIZE = 32


def update_progress(progress, opt_text=""):
    """
    Print a rewritable single line ASCI progress bar.
    :param progress: Progress from 0.0 to 1.0.
    """
    bar_length = 20
    block = int(math.ceil(bar_length * progress))
    print("Progress: [{}] {:.1f}%   {:s}".format( "#" * block + "-" * (bar_length - block),
                                                 progress * 100, opt_text), end='\r')

def parse_driving_logs(flip=True, cameras=[0,1,2]):

    if not os.path.exists(data_dir):
        return

    if flip and not os.path.exists(img_flip_dir):
        os.mkdir(img_flip_dir)

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
                    img_path = img_dir + '/' + basename

                    if not os.path.isfile(img_path):
                        continue

                    angle = float(line[3])
                    # apply steering correction angle to left and right camera pictures
                    if i == 1:
                        angle = angle + STEERING_CORRECTION
                    elif i == 2:
                        angle = angle - STEERING_CORRECTION

                    samples.append((img_path, angle))

                    # generate a horizontally flipped variant of the image
                    if flip:
                        img_flip_path = img_flip_dir + '/' + basename
                        if not os.path.isfile(img_flip_path):
                            img_rgb = Image.open(img_path)
                            img_rgb_flip = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                            img_rgb_flip.save(img_flip_path)

                        samples.append((img_flip_path, -angle))

    return samples


def create_model(input_shape=(160, 320, 3),
                   vertical_cropping=(74, 20),
                   dropout_rate=0.2):

    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=(vertical_cropping, (0, 0))))

    model.add(Conv2D(24, 5, strides=(2, 3), padding='valid', activation='relu'))
    model.add(Conv2D(36, 5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(48, 5, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dropout(dropout_rate))
    model.add(Dense(100))

    model.add(Dropout(dropout_rate))
    model.add(Dense(50))

    model.add(Dropout(dropout_rate))
    model.add(Dense(10))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    return model


def plot_history(history_object):
    """
    Plot the training and testing loss for each epoch in the provided history.
    """

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def normalize_image(img):
    col_min, col_max = np.min(img), np.max(img)
    return (img - col_min) / (col_max - col_min)


def main():
    samples = parse_driving_logs()
    df = pd.DataFrame(samples, columns=['filename', 'angle'])

    df_train, df_valid = train_test_split(df, shuffle=True, test_size=0.2)

    datagen_train = ImageDataGenerator(
        # rescale=1.0/255.0,
        rotation_range=3.0,
        shear_range=3.0,
        zoom_range=0.03,
        fill_mode='nearest')
    train_generator = datagen_train.flow_from_dataframe(dataframe=df_train, x_col='filename', y_col='angle',
        class_mode='raw', target_size=(160, 320), batch_size=BATCH_SIZE,
        interpolation='bilinear')

    datagen_valid = ImageDataGenerator()
    valid_generator = datagen_valid.flow_from_dataframe(dataframe=df_valid, x_col='filename', y_col='angle',
        class_mode='raw', target_size=(160, 320), batch_size=BATCH_SIZE,
        interpolation='bilinear')

    model = create_model()
    model.summary()
    plot_model(model, to_file=data_dir + '/model.png',
               show_shapes=True, show_layer_names=True, rankdir="TB", expand_nested=False, dpi=96)

    # compile and train the model using the generator function
    model.compile(loss='mse', optimizer='adam')

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size

    # batch = next(train_generator)
    # for i in range(5):
    #     plt.imshow(batch[0][i])
    #     plt.show()
    #
    # batch = next(valid_generator)
    # for i in range(5):
    #     plt.imshow(batch[0][i])
    #     plt.show()

    history = model.fit(
        train_generator,
        steps_per_epoch=step_size_train,
        validation_data=valid_generator,
        validation_steps=step_size_valid,
        workers=os.cpu_count()//2,
        # use_multiprocessing=True,
        shuffle=True,
        epochs=NB_EPOCHS,
        verbose=1)

    model.save(data_dir + '/model.h5')

    print("Model saved.")

    plot_history(history)


    # for i in range(3):
    #     test_samples = parse_driving_logs(False, [i])
    #     test_images = []
    #     for fn, angle in test_samples:
    #         test_images.append(np.asarray(Image.open(fn)))
    #     test_images = np.asarray(test_images)
    #     predictions = model.predict(test_images)
    #     print("{}: {}".format(i, np.nanmean(predictions)))


if __name__ == "__main__":
    main()
