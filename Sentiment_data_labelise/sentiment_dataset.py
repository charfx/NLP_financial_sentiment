from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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

    Stratification sur le label afin de conserver
    approximativement la même distribution des classes.
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


def compute_training_class_weights(
    train_df: pd.DataFrame,
) -> torch.Tensor:
    """
    Calcule les poids de classes uniquement à partir
    du jeu d'entraînement.

    Labels :
    0 = Negative
    1 = Neutral
    2 = Positive
    """

    classes = np.array([0, 1, 2])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return class_weights


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
    Charge les données,
    crée les splits,
    calcule les poids des classes,
    configure le tokenizer,
    puis construit les DataLoaders PyTorch.

    Retour :
    - train_loader
    - validation_loader
    - test_loader
    - class_weights
    """

    # ======================================================
    # LOAD DATA
    # ======================================================

    dataframe = pd.read_csv(
        DATA_PATH
    )

    # ======================================================
    # SPLITS
    # ======================================================

    train_df, validation_df, test_df = (
        create_splits(dataframe)
    )

    # ======================================================
    # CLASS WEIGHTS
    # ======================================================

    class_weights = (
        compute_training_class_weights(
            train_df
        )
    )

    # ======================================================
    # TOKENIZER
    # ======================================================

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    pad_id = tokenizer.token_to_id(
        "[PAD]"
    )

    if pad_id is None:
        raise ValueError(
            "Le token [PAD] est absent du tokenizer."
        )

    tokenizer.enable_truncation(
        max_length=MAX_LENGTH
    )

    tokenizer.enable_padding(
        length=MAX_LENGTH,
        pad_id=pad_id,
        pad_token="[PAD]",
    )

    # ======================================================
    # DATASETS
    # ======================================================

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

    # ======================================================
    # DATALOADERS
    # ======================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_weights,
    )