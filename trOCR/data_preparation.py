import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import IAMDataset
from transformers import TrOCRProcessor


def load_and_preprocess_data(csv_file='train_labels.csv', root_dir='all_images/'):
    """
    loading the csv of the labels , preprocess it and split it to train and evaluation sets, then it loads the processor
    and creates a dataset generator.
    returns the generator objects and the processor.
    """
    df = pd.read_csv(csv_file, sep='\t', header=None, skiprows=1)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    df['file_name'] = df['file_name'].apply(lambda x: x + '.png')

    train_df, val_df = train_test_split(df, test_size=0.2)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')

    train_dataset = IAMDataset(root_dir, train_df, processor)
    eval_dataset = IAMDataset(root_dir, val_df, processor)

    return train_dataset, eval_dataset, processor
