# Handwriting-Recognition
<img width="300" src="https://www.comidor.com/wp-content/uploads/2022/08/ocr-55-e1661521818617-1024x569.png" alt="nasdaq_image">

OCR stands for Optical Character Recognition. It is a technology used to convert different types of documents, such as scanned paper documents, PDF files or images captured by a digital camera, into editable and searchable data.
OCR has a wide range of applications and is used to automate data extraction and to improve the efficiency of data processing in numerous industries.

## Dataset:
[IAM Handwriting Database Link](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

The IAM Handwriting Database is a publicly accessible and freely available dataset that is widely used for research purposes. It contains handwritten forms, text lines, sentences, and words, providing a rich resource for studying handwriting recognition, optical character recognition (OCR), and related fields.

The IAM Handwriting Database is hierarchically structured into different categories. The dataset includes the following components:

* Forms: This category contains form images, where each image represents a complete handwritten form. The forms are named following the LOB Corpus naming scheme, such as “a01-122.png”.


* Text Lines: The text lines category contains individual lines of text extracted from the form images. Each line is saved as a separate image, following a similar naming convention as the forms.


* Sentences: This category contains sentences, with one sentence corresponding to each text line image. The sentences provide additional context and can be used for higher-level analysis.


* Words: This category includes individual words extracted from the text lines. Similar to text lines, each word is saved as an image.


* ASCII and XML: These categories contain meta-information about the forms, lines, sentences, and words in ASCII and XML formats, respectively. They provide summarized information about the dataset, including labels, coordinates, and other relevant details.


## Repository Structure:

1. MS2 - Contains files for milestone 2:
    * Baseline Model
    * config - configuration file for baseline model
    * Images EDA 
    * Labels EDA
    * Preprocessing - Image preprocessing
   

2. MS3 - Contains files for milestone 3:
   * TrOCR transformers model fine-tuning 
   * Donut transformers model fine-tuning
   * CNN-Transformers model
   * resize - images directory after preprocessing
   * labels.csv - the labels file after preprocessing.

## Running the file:

Most files are notebooks with hardcoded predefined configuration.
The only file that requires instructions to run is the preprocessing.py file in MS2 since we wanted to keep the image preprocessing as dynamic as possible.

Below is an example of how we ran the preprocessing.py file to create the processed images we used to feed the different models:

    python preprocessing.py --batch -input_folder all_images -dest_folder resize -new_height 64 -new_width 512