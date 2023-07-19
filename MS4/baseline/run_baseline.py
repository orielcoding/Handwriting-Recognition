import os

from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image

import train
import evaluate

path_to_data = "/Users/mathias/ITC/[PROJECT]/Final/data/preprocessed_data"
path_to_preprocessed_data = "/Users/mathias/ITC/[PROJECT]/Final/data/preprocessed_data"

# path_to_data = "/Users/mathias/ITC/[PROJECT]/Final/data/mini_preprocessed_data"


def main():

    # arguments : if we want to train ; predict ; evaluate
    # arguments to preprocess or not

    # path_to_preprocessed_data =

    train_images, test_images, train_labels, test_labels = load_data(path_to_data)

    vocabulary = get_vocabulary(train_labels)
    max_sequence_length = get_max_seq_length(train_labels)
    train_labels_preprocessed = label_preprocessing(train_labels, vocabulary)

    history = train_baseline(train_images, train_labels_preprocessed, max_sequence_length)

    # evaluate_baseline(test_images, test_labels, vocabulary)



def train_baseline(train_images, train_labels_preprocessed, max_length):
    """

    :param train_images:
    :param train_labels_preprocessed:
    :param max_length:
    :return:
    """
    history = train.train(train_images, train_labels_preprocessed, max_length)
    return history


def evaluate_baseline(test_images, test_labels, vocabulary):
    """

    :param test_images:
    :param test_labels:
    :param vocabulary:
    :return:
    """
    cer, wer = evaluate.baseline_evaluate(test_images, test_labels, vocabulary)
    print('CER mean: ', cer)
    print('WER mean: ', wer)



def predict(path_to_prediction):
    """

    :return:
    """

    images_to_predict = load_images(path_to_prediction)

    baseline_model = keras.models.load_model('saved_model/baseline_model')
    preds = baseline_model.predict(images_to_predict)


def load_data(path_to_data):

    path_to_train_images = path_to_data + "/train_test/train"
    path_to_train_labels = path_to_data + "/train_labels.csv"

    path_to_test_images = path_to_data + "/train_test/test"
    path_to_test_labels = path_to_data + "/test_labels.csv"

    train_images = load_images(path_to_train_images)
    train_labels = load_labels(path_to_train_labels)

    test_images = load_images(path_to_test_images)
    test_labels = load_labels(path_to_test_labels)

    return train_images, test_images, train_labels, test_labels


def load_labels(labels_path):
    labels_head = pd.read_csv(labels_path, header=None)
    labels = labels_head.iloc[:, 1].to_numpy()
    return labels


def load_images(folder_path):
    """

    :param folder_path:
    :return:
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

    :param labels_sentence:
    :return:
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
    creates a vocabulary for the translation of the prediction
    """
    vocabulary = sorted(set(''.join(train_labels)))
    vocabulary_dict = {char: index for index, char in enumerate(vocabulary)}
    return vocabulary_dict




if __name__ == '__main__':
    main()
