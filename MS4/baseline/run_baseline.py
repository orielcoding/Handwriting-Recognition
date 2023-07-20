import os

from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
import subprocess
import json

import train
import evaluate

# loading configuration file
with open("config.json") as f:
    config = json.load(f)


def main():

    # arguments : if we want to train ; predict ; evaluate
    # arguments to preprocess or not

    train_images, test_images, train_labels, test_labels = load_data(config["path_to_preprocessed_data"])

    vocabulary = get_vocabulary(train_labels)
    max_sequence_length = get_max_seq_length(train_labels)
    train_labels_preprocessed = label_preprocessing(train_labels, vocabulary)

    history = train_baseline(train_images, train_labels_preprocessed, max_sequence_length)

    # evaluate_baseline(test_images, test_labels, vocabulary)


def train_baseline(train_images, train_labels_preprocessed, max_length):
    """
    calls the function that train the model
    :param train_images: images train set
    :param train_labels_preprocessed: images' labels for training
    :param max_length: the length of the biggest sentence
    :return: the history of the model's training
    """
    history = train.train(train_images, train_labels_preprocessed, max_length)
    return history


def evaluate_baseline(test_images, test_labels, vocabulary):
    """

    :param test_images: images test set
    :param test_labels: images' labels for testing
    :param vocabulary: all the characters present in the train set
    :return: cer and wer (character and word error rates) floats, the two metrics of the models
    """
    cer, wer = evaluate.baseline_evaluate(test_images, test_labels, vocabulary)
    print('CER mean: ', cer)
    print('WER mean: ', wer)



def predict(path_to_prediction):
    """
    performs the reading on a preprocessed image not present in the data set
    :path_to_prediction: path to the preprocessed image to read
    :return: the sentence the model read from the input images
    """

    images_to_predict = load_images(path_to_prediction)

    baseline_model = keras.models.load_model('saved_model/baseline_model')
    predicted_label = baseline_model.predict(images_to_predict)


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


if __name__ == '__main__':
    main()
