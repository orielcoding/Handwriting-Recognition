import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import string
import os

def predict_ocr(image):
    """
    Accepts image and inference from the pre-trained TrOCR model,
    """
    # Call the preprocessing python file and set it for the image
    image = Image.fromarray(image.astype('uint8'), 'RGB')

    # Predict the sentence - use
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    # Map the predicted label to the corresponding character
    characters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    predicted_character = characters[predicted_label]

    return predicted_character

def main():
    model = load_model("")  # load jons model, should be like : path/to/your/ocr/model.h5

    # Define the Gradio interface
    inputs = gr.inputs.Image()
    outputs = gr.outputs.Textbox()
    gr.Interface(fn=predict_ocr, inputs=gr.Image(type='filepath'), outputs=gr.Textbox(), title="OCR Demo").launch()

if __name__ == "__main__":
    main()
