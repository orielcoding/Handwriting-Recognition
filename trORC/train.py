from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from data_preparation import load_and_preprocess_data
from model import get_model
from datasets import load_metric


def compute_metrics(processor, pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer_metric = load_metric("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def main():
    train_dataset, eval_dataset, processor = load_and_preprocess_data()
    model = get_model(processor)

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
        compute_metrics=lambda pred: compute_metrics(processor, pred),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.model.save_pretrained('./saved_model/')


if __name__ == '__main__':
    main()
