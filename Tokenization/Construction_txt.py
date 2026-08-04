## Article clean pret passée par deux filtre de cleaning
## lecture colone headline + article 
##construction title article format txt claire pour prochain tokenization

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Headline",
    "Article",
}


def normalize_text_value(value: object) -> str:
    """
    Convertit une valeur issue du DataFrame en texte propre.

    - NaN ou None deviennent une chaîne vide.
    - Les espaces multiples sont normalisés.
    - Les retours à la ligne utiles restent présents.
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value)

    lines = [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines).strip()


def build_article_text(
    headline: object,
    article: object,
) -> str:
    """
    Construit le texte qui sera transmis au tokenizer.

    Exemple :
    [TITLE] NVIDIA Raises Revenue Forecast
    [ARTICLE] Demand for AI chips remains strong.
    """
    clean_headline = normalize_text_value(headline)
    clean_article = normalize_text_value(article)

    parts = []

    if clean_headline:
        parts.append(f"[TITLE] {clean_headline}")

    if clean_article:
        parts.append(f"[ARTICLE] {clean_article}")

    return "\n".join(parts).strip()


def validate_corpus_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Vérifie que les colonnes indispensables existent.
    """
    missing_columns = REQUIRED_COLUMNS.difference(
        dataframe.columns
    )

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes dans le corpus : "
            f"{sorted(missing_columns)}"
        )


def load_clean_corpus(
    parquet_path: str | Path,
) -> pd.DataFrame:
    """
    Charge le corpus final nettoyé depuis le fichier Parquet.
    """
    path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {path.resolve()}"
        )

    dataframe = pd.read_parquet(
        path,
        columns=["Headline", "Article"],
    )

    validate_corpus_columns(dataframe)

    return dataframe


def article_text_iterator(
    parquet_path: str | Path,
) -> Iterator[str]:
    """
    Produit un article à la fois sous forme de texte.

    L'utilisation de yield évite de construire une liste
    contenant les 441 626 articles en mémoire.
    """
    dataframe = load_clean_corpus(parquet_path)

    for row in dataframe.itertuples(index=False):
        text = build_article_text(
            headline=row.Headline,
            article=row.Article,
        )

        if text:
            yield text


def preview_corpus(
    parquet_path: str | Path,
    number_of_articles: int = 3,
) -> None:
    """
    Affiche quelques textes tels qu'ils seront reçus
    par le futur tokenizer.
    """
    if number_of_articles < 1:
        raise ValueError(
            "number_of_articles doit être supérieur à zéro."
        )

    iterator = article_text_iterator(parquet_path)

    for article_number, text in enumerate(
        iterator,
        start=1,
    ):
        print("\n" + "=" * 100)
        print(f"ARTICLE PRÉPARÉ N° {article_number}")
        print("=" * 100)
        print(text[:2_000])

        if article_number >= number_of_articles:
            break
