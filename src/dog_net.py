import logging
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump, load
from sklearn.preprocessing import OneHotEncoder

import src.logging as dog_net_logging

dog_net_logging.setup_logging()

logger = logging.getLogger(__name__)

PATH_MODEL = './model/dog_net.h5'
PATH_TRAIN_DATA = "./resources/train/"
PATH_ONE_HOT_ENCODER = "./encoder/ohe.joblib"
PATH_TRAINED = "./resources/trained.txt"
PATH_LABELS = "./resources/labels.csv"

LABEL_INDEX = 1
COLOR_DEPTH = 3  # RGB

if os.path.isfile(PATH_ONE_HOT_ENCODER):
    one_hot_encoder = load(PATH_ONE_HOT_ENCODER)
    logger.info("Existing one hot encoder loaded successfully.")
else:
    one_hot_encoder = OneHotEncoder(sparse=False)
    logger.info("New one hot encoder created successfully.")

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.BatchNormalization(input_shape=(None, None, COLOR_DEPTH)))
model.add(tf.keras.layers.Conv2D(filters=300,
                                 kernel_size=(2, 2),
                                 strides=(1, 1),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Conv2D(filters=200, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
model.add(tf.keras.layers.GlobalMaxPool2D())
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=300, activation='relu'))
model.add(tf.keras.layers.Dense(units=120, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


def load_model():
    logger.info("Loading model from: " + PATH_MODEL)
    global model
    model = tf.keras.models.load_model(filepath=PATH_MODEL)
    model.summary()
    logger.info("Model loaded successfully.")


def predict(x):
    '''
    Predicts a dog breed from image.
    :param x: Numpy array with shape (None, None, 3) representing a colored image
    :return: tuple (confidence, prediction)
    confidence: representing how confident the network is about each dog breed in percentage
    prediction: the predicted dog breed
    '''
    logger.debug("Predicting.")
    x = np.expand_dims(x, axis=0)  # batching 1
    result = model.predict(x=x, batch_size=1)
    decoded_result = one_hot_encoder.inverse_transform(result)[0][0]
    logger.debug("Prediction confidence:\n" + str(result[0]))
    logger.debug("Prediction: " + str(decoded_result))
    return result[0], decoded_result


def predict_from_file(path):
    logger.info("Predicting from file: " + path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return predict(img)


def evaluate(x, y, batch_size=32):
    score = model.evaluate(x=x, y=y, batch_size=batch_size)
    logging.info("Evaluation:\t" + score[0])
    return score


def train(x, y, batch_size=4, epochs=10):
    model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, validation_split=0.3, shuffle=True)
    logger.debug("Training finished.")
    logger.info("Saving model to: " + str(PATH_MODEL))
    tf.keras.models.save_model(model=model, filepath=PATH_MODEL, overwrite=True, include_optimizer=True)
    logger.info("Model saved successfully.")


def train_from_files(data_batch_size=8, training_batch_size=4, epochs=10):
    num_of_training_files = len(os.listdir(PATH_TRAIN_DATA))
    while num_of_training_files > 0:
        data, labels = load_data(batch_size=data_batch_size)
        train(data, labels, batch_size=training_batch_size, epochs=epochs)
        num_of_training_files = num_of_training_files - data_batch_size


def load_data(batch_size=8):
    logger.info("Loading data from: " + PATH_TRAIN_DATA)
    data = []
    labels = []
    max_w = 0
    max_h = 0
    counter = 0
    labels_df = pd.read_csv(PATH_LABELS, skiprows=1, header=None)
    all_possible_labels = labels_df[labels_df.columns[-1]].values.reshape(-1, 1)
    one_hot_encoder.fit(all_possible_labels)
    labels_matrix = labels_df.values
    for filename in (os.listdir(PATH_TRAIN_DATA)):
        full_path_filename = os.path.join(PATH_TRAIN_DATA, filename)

        already_trained_file_reader = open(PATH_TRAINED, mode='r')
        trained = filename in already_trained_file_reader.read()
        already_trained_file_reader.close()

        is_file = os.path.isfile(full_path_filename)
        if trained:
            logger.debug("Skipping file: " + full_path_filename)
        elif is_file and counter < batch_size:
            filename_without_extension = filename.split('.')[0]
            img_index, _ = np.where(labels_matrix == filename_without_extension)
            label = labels_matrix[img_index][0][LABEL_INDEX]
            labels.append(label)
            logger.debug("Processing: " + full_path_filename)
            img = cv2.imread(full_path_filename, cv2.IMREAD_COLOR)
            img_w = img.shape[0]
            img_h = img.shape[1]
            if img_w > max_w:
                max_w = img_w
            if img_h > max_h:
                max_h = img_h
            data.append(img)

            already_trained_file_writer = open(PATH_TRAINED, mode='a')
            already_trained_file_writer.write(filename + '\n')
            already_trained_file_writer.close()

            counter = counter + 1

    data = _pad_data(max_h, max_w, data)
    labels = _one_hot_encode_labels(labels)
    logger.info("Data loaded successfully.")
    return data, labels


def _pad_data(max_h, max_w, data):
    logger.debug("Padding data.")
    padded_data = []
    padded_tmp = np.zeros((max_w, max_h, COLOR_DEPTH))
    for dat in data:
        padded_tmp[:dat.shape[0], :dat.shape[1], :dat.shape[2]] = dat
        padded_data.append(padded_tmp)
        padded_tmp = np.zeros((max_w, max_h, COLOR_DEPTH))
    padded_data = np.asarray(padded_data)
    return padded_data


def _one_hot_encode_labels(labels):
    logger.debug("One hot encoding labels.")
    labels = np.asarray(labels).reshape(-1, 1)
    labels = one_hot_encoder.transform(labels)
    logger.debug("Saving one hot encoder to: " + PATH_ONE_HOT_ENCODER)
    dump(one_hot_encoder, PATH_ONE_HOT_ENCODER)
    logger.debug("One hot encoder saved successfully.")
    return labels
