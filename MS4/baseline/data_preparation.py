import os

from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
import subprocess
import json


# loading configuration file
with open("config.json") as f:
    config = json.load(f)


def load_data(path_to_data):
    """
    loads the data that will be read by the model
    :param path_to_data:
    :return: the labels and images splited as train and test
    """

    path_to_train_images = path_to_data + "/train"
    path_to_train_labels = path_to_data + "/train_labels.csv"

    path_to_test_images = path_to_data + "/test"
    path_to_test_labels = path_to_data + "/test_labels.csv"

    train_images = load_images(path_to_train_images)
    train_labels = load_labels(path_to_train_labels)

    test_images = load_images(path_to_test_images)
    test_labels = load_labels(path_to_test_labels)

    return train_images, test_images, train_labels, test_labels


def load_labels(labels_path):
    """
    load labels from the csv file into an array of string
    :param labels_path: path to the csv file containing the labels
    :return: a numpy array containing all the labels
    """
    labels_head = pd.read_csv(labels_path, header=None)
    labels = labels_head.iloc[:, 1].to_numpy()
    return labels


def load_images(folder_path):
    """
    loading the preprocessed images as numpy arrays
    :param folder_path: a path to the folder containing the images
    :return: a numpy array containing the preprocessed images a numpy arrays
    """
    images = []
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image = np.array(Image.open(image_path))
        images.append(image)

    images = np.array(images)
    return images


def get_max_seq_length(labels_sentence):
    """
    gets the length of the longest sentence
    :param labels_sentence: numpy arrays containing the labels
    :return: the length of the longest sentence
    """
    return max(len(sentence) for sentence in labels_sentence)


def label_preprocessing(labels_sentence, vocabulary):
    """
    Takes in a single label as a string (sentence matching the content of the image) and
    preprocesses it so that the label can interpret by the model
    Param: label_sentence: the label a string
    Returns: The preprocessed label as a sequence of indexes
    """

    max_seq_length = get_max_seq_length(labels_sentence)

    labels_indexes = [[vocabulary[char] for char in sentence] for sentence in labels_sentence]
    preprocessed_label = keras.preprocessing.sequence.pad_sequences(labels_indexes, maxlen=max_seq_length, padding='post')

    return preprocessed_label


def get_vocabulary(train_labels):
    """
    creates a mapping of the prediction into a vocabulary
    :param train_labels: a list the labels as they were read from the csv file
    :return: a dictionnary mapping an index with any character present in the train labels
    """
    vocabulary = sorted(set(''.join(train_labels)))
    vocabulary_dict = {char: index for index, char in enumerate(vocabulary)}
    return vocabulary_dict


def preprocess_data(commmand):
    """

    :param commmand:
    :return:
    """

    # Run the bash command using subprocess
    try:
        completed_process = subprocess.run(commmand, shell=True, check=True)
        # If the command runs successfully, the completed_process object will contain the result
        print("Bash command executed successfully.")
    except subprocess.CalledProcessError as e:
        # If there is an error running the command, it will raise a CalledProcessError
        print(f"Error executing bash command: {e}")



