import json
from tensorflow import keras

# loading configuration file
with open("config.json") as f:
    config = json.load(f)


def create_baseline(input_dim_1, input_dim_2, max_sequence_length):
    """

    :param input_dim_1:
    :param input_dim_2:
    :param max_sequence_length:
    :return:
    """

    # Hyperparameters:
    input_shape = (input_dim_1,
                   input_dim_2,
                   1)

    optimizer = keras.optimizers.Adam(learning_rate=config["LR"])
    loss = keras.losses.SparseCategoricalCrossentropy()

    # Define the CNN-GRU model
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.RepeatVector(max_sequence_length))
    model.add(keras.layers.GRU(128, return_sequences=True, kernel_initializer='he_normal', name='gru1'))
    model.add(keras.layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b'))
    model.add(keras.layers.Dense(80, activation='softmax', kernel_initializer='he_normal', name='dense2'))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=config["METRIC"])

    return model
