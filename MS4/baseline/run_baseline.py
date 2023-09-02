from tensorflow import keras
import json

import train
import evaluate
import data_preparation

# loading configuration file
with open("config.json") as f:
    config = json.load(f)


def main():

    # train_images, test_images, train_labels, test_labels = data_preparation.load_data(config["path_to_preprocessed_data"])
    #
    # vocabulary = data_preparation.get_vocabulary(train_labels)
    # max_sequence_length = data_preparation.get_max_seq_length(train_labels)
    # train_labels_preprocessed = data_preparation.label_preprocessing(train_labels, vocabulary)

    # history = train_baseline(train_images, train_labels_preprocessed, max_sequence_length)
    #
    # evaluate_baseline(test_images, test_labels, vocabulary)

    predicted_label = predict_baseline(config["path_to_prediction_image"])

    print(predicted_label)


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


def predict_baseline(path_to_prediction):
    """
    performs the reading on a preprocessed image not present in the data set
    :path_to_prediction: path to the preprocessed image to read
    :return: the sentence the model read from the input images
    """

    images_to_predict = data_preparation.load_images(path_to_prediction)

    baseline_model = keras.models.load_model(config["path_to_model"] + "baseline_model")
    predicted_label = baseline_model.predict(images_to_predict)


if __name__ == '__main__':
    main()
