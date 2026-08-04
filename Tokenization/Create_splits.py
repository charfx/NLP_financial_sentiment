from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_tokenizer_splits() -> None:
    source_path = Path(
        "data/processed/articles_cleaned_final.parquet"
    )

    output_directory = Path(
        "data/splits"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Corpus final introuvable : {source_path.resolve()}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Chargement du corpus final...")

    dataframe = pd.read_parquet(
        source_path,
        columns=["Headline", "Article"],
    )

    print(f"Articles disponibles : {len(dataframe):,}")

    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    train_path = (
        output_directory / "tokenizer_train.parquet"
    )

    validation_path = (
        output_directory / "tokenizer_validation.parquet"
    )

    test_path = (
        output_directory / "tokenizer_test.parquet"
    )

    train_df.to_parquet(
        train_path,
        index=False,
    )

    validation_df.to_parquet(
        validation_path,
        index=False,
    )

    test_df.to_parquet(
        test_path,
        index=False,
    )

    print("\n=== TOKENIZER SPLITS CREATED ===")
    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(validation_df):,}")
    print(f"Test       : {len(test_df):,}")

    print("\nFichiers créés :")
    print(train_path)
    print(validation_path)
    print(test_path)


if __name__ == "__main__":
    create_tokenizer_splits()