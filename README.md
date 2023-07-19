# Handwriting-Recognition

In this repository, we present our fine-tuned TrOCR model for the text lines dataset from the IAM handwriting database. The IAM is publicly accessible and freely available. This dataset contains a general type of handwritten documents and with the fine-tuned model for it, you can use our implementation to turn documents into machine-readable format.     

The purpose of this repository is to suggest a possible fine-tuning for general OCR models. 

The TrOCR directory contains several .py files and a configuration file. To run the model:
1. Download the TrOCR directory 
2. Install the requirements.txt file. 
3. If desired, change the settings of the training through the confing.json file. 
4. Run the 'train.py'. The model will be saved to a file called 'saved_model' in the directory to which you downloaded the TrOCR directory.
5. Run the 'predict.py' file either from the terminal, calling the 'predict' function and enter the file path to an image you would like to convert to machine-readable format.

# IAM Database
[IAM Database Site](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)


 # Authors
[Jonathan Schwarz](https://www.linkedin.com/in/jonathan-schwarz91?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BKYHvdFIYT1y0vj4pBscFfg%3D%3D)

[Oriel Singer](https://www.linkedin.com/in/oriel-singer/)

[Mathias Kammoun](https://www.linkedin.com/in/mathias-kammoun?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BZsEpQGnsRRWKpnYLJuyAwA%3D%3D)

Tzaji Minuchin
