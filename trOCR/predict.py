import torch
from dataset import IAMDataset
from transformers import VisionEncoderDecoderModel
import pandas as pd
import os
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from transformers import TrOCRProcessor
import gradio as gr


def predict(image_path):

    path = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    df_test = pd.DataFrame()
    df_test.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    df_test = df_test.append({'file_name': file_name, 'text': '0'}, ignore_index=True)

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
    dataset = IAMDataset(root_dir=path, df=df_test, processor=processor)

    model = VisionEncoderDecoderModel.from_pretrained("./my_model/content/my_model")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        output_dir="./",
        logging_steps=2,
        save_steps=1000,
        eval_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        data_collator=default_data_collator,
    )

    pred = trainer.predict(dataset)
    pred_ids = pred.predictions
    print(f"The predicted text: {processor.batch_decode(pred_ids, skip_special_tokens=True)[0]}")

# gr.Interface(fn=git log --all --decorate --oneline --graphpredict_ocr, inputs=gr.Image(type='filepath'), outputs=gr.Textbox(), title="OCR Demo").launch()
print(predict('3.jpeg'))
