from tensorflow import keras
import difflib
from jiwer import wer
import json
import numpy as np


# loading configuration file
with open("config.json") as f:
    config = json.load(f)

baseline_model = keras.models.load_model('saved_model/baseline_model')

def word_error_rate(y_true, y_pred):
    """
    returns calculation of Word Error Rate for a prediction.
    """
    wer_value = wer(y_true, y_pred)
    return wer_value


def character_error_rate(y_true, y_pred):
    """
    returns calculation of Character Error Rate for a prediction.
    """
    matcher = difflib.SequenceMatcher(None, y_true, y_pred)
    cer_value = 1 - matcher.ratio()
    return cer_value


def baseline_evaluate(test_images, test_labels, vocabulary):
    """

    :param test_images:
    :param test_labels:
    :param vocabulary:
    :return:
    """

    preds = baseline_model.predict(test_images)
    index_to_char = {v: k for k, v in vocabulary.items()}
    cer_sum = 0
    wer_sum = 0

    for i, pred in enumerate(preds):
        indices = np.argmax(pred, axis=-1)

        characters = ''.join([index_to_char[idx] for idx in indices])
        cer_sum += character_error_rate(test_labels[i], characters)
        wer_sum += word_error_rate(test_labels[i], characters)

    avg_CER = cer_sum / len(test_labels)
    avg_WER = wer_sum / len(test_labels)

    print(avg_CER, avg_WER)

    return avg_CER, avg_WER

