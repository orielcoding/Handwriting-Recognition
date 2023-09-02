import json
import model
import os

from tensorflow import keras

# loading configuration file
with open("config.json") as f:
    config = json.load(f)


def train(train_images, train_labels_preprocessed, max_sequence_length):
    """
    creates and trains the model
    :param train_images: images preprocessed for training
    :param train_labels_preprocessed: labels preprocessed for training
    :param max_sequence_length: the length of the longest sentence of the train set
    :return: history of the training
    """

    input_dim_1 = train_images.shape[1]
    input_dim_2 = train_images.shape[2]

    baseline_model = model.create_baseline(input_dim_1, input_dim_2, max_sequence_length)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = baseline_model.fit(train_images, train_labels_preprocessed,
                                 validation_split=config["VAL_SPLIT"],
                                 epochs=config["EPOCHS"],
                                 batch_size=config["BATCH_SIZE"],
                                 callbacks=[callback])


    # checking if the directory demo_folder exist or not.
    if not os.path.exists(config["path_to_model"]):
        # if the demo_folder directory is not present then create it.
        os.makedirs(config["path_to_model"])

    model_path = config["path_to_model"] + 'baseline_model'
    baseline_model.save(model_path)
    print(f'model was successfully saved in {model_path}')

    return history


if __name__ == '__main__':
    train()

