from pathlib import Path

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


TOKENIZER_PATH = Path(
    "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
)

DATA_PATH = Path(
    "Sentiment_data_labelise/data_cleaned/sentiment_cleaned.csv"
)

MAX_LENGTH = 512


def create_splits(
    dataframe: pd.DataFrame,
    random_state: int = 42,
):
    """
    Split :
    80% train
    10% validation
    10% test
    """

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=random_state,
        stratify=dataframe["label"],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class FinancialSentimentDataset(Dataset):
    """
    Transforme chaque phrase financière en :

    input_ids
    attention_mask
    label
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Tokenizer,
    ) -> None:
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        row = self.dataframe.iloc[index]

        sentence = str(
            row["Sentence"]
        )

        label = int(
            row["label"]
        )

        encoding = self.tokenizer.encode(
            sentence
        )

        input_ids = torch.tensor(
            encoding.ids,
            dtype=torch.long,
        )

        attention_mask = torch.tensor(
            encoding.attention_mask,
            dtype=torch.long,
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label_tensor,
        }


def build_dataloaders(
    batch_size: int = 16,
):
    """
    Charge les données, crée les splits,
    configure le tokenizer puis construit
    les DataLoaders PyTorch.
    """

    dataframe = pd.read_csv(
        DATA_PATH
    )

    train_df, validation_df, test_df = (
        create_splits(dataframe)
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    pad_id = tokenizer.token_to_id(
        "[PAD]"
    )

    tokenizer.enable_truncation(
        max_length=MAX_LENGTH
    )

    tokenizer.enable_padding(
        length=MAX_LENGTH,
        pad_id=pad_id,
        pad_token="[PAD]",
    )

    train_dataset = FinancialSentimentDataset(
        train_df,
        tokenizer,
    )

    validation_dataset = FinancialSentimentDataset(
        validation_df,
        tokenizer,
    )

    test_dataset = FinancialSentimentDataset(
        test_df,
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )