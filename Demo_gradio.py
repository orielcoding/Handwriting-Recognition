import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import string
import os

# Load the pre-trained fine tuned TrOCR model
model = load_model("")  # load jons model, should be like : path/to/your/ocr/model.h5

# Define the directory to save the images and predictions
save_directory = "path/to/save/directory"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)


def predict_ocr(image):
    # Call the preprocessing python file and set it for the image
    image = Image.fromarray(image.astype('uint8'), 'RGB')

    # Predict the sentence - use
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    # Map the predicted label to the corresponding character
    characters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    predicted_character = characters[predicted_label]

    # Save the input image and prediction
    input_image_path = os.path.join(save_directory, "input_image.jpg")
    output_prediction_path = os.path.join(save_directory, "output_prediction.txt")

    image.save(input_image_path)
    with open(output_prediction_path, "w") as f:
        f.write(predicted_character)

    return predicted_character


def main():
    # Define the Gradio interface
    inputs = gr.inputs.Image()
    outputs = gr.outputs.Textbox()
    gr.Interface(fn=predict_ocr, inputs=inputs, outputs=outputs, title="OCR Demo").launch()


if __name__ == "__main__":
    main()
