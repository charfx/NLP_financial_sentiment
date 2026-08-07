from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path(
    "Sentiment_data_labelise/data_not_cleaned/data2.csv"
)

CLEAN_DATA_DIR = Path(
    "Sentiment_data_labelise/data_cleaned"
)

CLEAN_DATA_PATH = CLEAN_DATA_DIR / "sentiment_cleaned.csv"


LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def clean_sentiment_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nettoyage léger du dataset supervisé de sentiment.
    """

    required_columns = {
        "Sentence",
        "Sentiment",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise KeyError(
            f"Colonnes manquantes : {missing_columns}"
        )

    cleaned_df = dataframe.copy()

    # ------------------------------------------------------
    # 1. Suppression des NaN éventuels
    # ------------------------------------------------------

    cleaned_df = cleaned_df.dropna(
        subset=["Sentence", "Sentiment"]
    )

    # ------------------------------------------------------
    # 2. Normalisation légère des textes
    # ------------------------------------------------------

    cleaned_df["Sentence"] = (
        cleaned_df["Sentence"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ------------------------------------------------------
    # 3. Suppression des textes vides
    # ------------------------------------------------------

    cleaned_df = cleaned_df[
        cleaned_df["Sentence"].str.len() > 0
    ]

    # ------------------------------------------------------
    # 4. Normalisation des labels
    # ------------------------------------------------------

    cleaned_df["Sentiment"] = (
        cleaned_df["Sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ------------------------------------------------------
    # 5. Vérification des labels autorisés
    # ------------------------------------------------------

    invalid_labels = set(
        cleaned_df["Sentiment"].unique()
    ).difference(
        LABEL_MAPPING.keys()
    )

    if invalid_labels:
        raise ValueError(
            f"Labels invalides détectés : {invalid_labels}"
        )

    # ------------------------------------------------------
    # 6. Suppression des doublons
    # ------------------------------------------------------

    cleaned_df = cleaned_df.drop_duplicates(
        subset=["Sentence", "Sentiment"]
    )

    # ------------------------------------------------------
    # 7. Encodage numérique
    # ------------------------------------------------------

    cleaned_df["label"] = (
        cleaned_df["Sentiment"]
        .map(LABEL_MAPPING)
        .astype(int)
    )

    cleaned_df.reset_index(
        drop=True,
        inplace=True,
    )

    return cleaned_df


def main() -> None:
    print("Chargement du dataset sentiment...")

    dataframe = pd.read_csv(
        RAW_DATA_PATH
    )

    print(f"Nombre initial : {len(dataframe):,}")

    print("\nDistribution initiale :")
    print(
        dataframe["Sentiment"]
        .value_counts()
    )

    cleaned_df = clean_sentiment_dataset(
        dataframe
    )

    print("\n=== CLEANING REPORT ===")

    print(
        f"Nombre final : {len(cleaned_df):,}"
    )

    print(
        f"Lignes supprimées : "
        f"{len(dataframe) - len(cleaned_df):,}"
    )

    print("\nDistribution finale :")
    print(
        cleaned_df["Sentiment"]
        .value_counts()
    )

    print("\nMapping labels :")
    print(LABEL_MAPPING)

    print("\nExemple :")
    print(
        cleaned_df[
            ["Sentence", "Sentiment", "label"]
        ].head()
    )

    # ------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------

    CLEAN_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_parquet(
        CLEAN_DATA_PATH,
        index=False,
    )

    print("\nDataset sauvegardé :")
    print(CLEAN_DATA_PATH)


if __name__ == "__main__":
    main()