import re
from typing import Tuple

import pandas as pd


# Dès qu'une signature Bloomberg est détectée,
# tout le contenu situé après celle-ci est supprimé.
BLOOMBERG_SIGNATURE_PATTERN = re.compile(
    r"\bTo contact the "
    r"(?:"
    r"reporter|reporters|"
    r"editor|editors|"
    r"news desk|news team"
    r")"
    r"(?: responsible for this story| on this story)?"
    r"\s*:.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def count_words(text: str) -> int:
    """
    Compte approximativement les mots d'un texte.

    Ce compteur sert au nettoyage initial.
    Le nombre réel de tokens sera calculé plus tard
    avec le tokenizer.
    """
    if not isinstance(text, str):
        return 0

    return len(text.split())


def remove_bloomberg_signature(text: str) -> str:
    """
    Supprime les signatures Bloomberg sans écraser
    les éventuels retours à la ligne existants.
    """
    if not isinstance(text, str):
        return ""

    cleaned_text = BLOOMBERG_SIGNATURE_PATTERN.sub("", text)

    # Uniformisation des retours à la ligne et espaces Unicode.
    cleaned_text = (
        cleaned_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
    )

    # Normalise seulement les espaces et tabulations
    # à l'intérieur de chaque ligne.
    cleaned_lines = []

    for line in cleaned_text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines).strip()


def clean_single_article(text: str) -> str:
    """
    Nettoie un article individuel.

    Cette fonction pourra être utilisée plus tard
    avec les nouveaux articles reçus depuis une API.
    """
    return remove_bloomberg_signature(text)


def apply_first_cleaning(
    dataframe: pd.DataFrame,
    article_column: str = "Article",
    min_words: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Premier nettoyage du corpus.

    Étapes :
    1. Suppression des signatures Bloomberg.
    2. Recalcul du nombre réel de mots.
    3. Rejet des articles contenant moins de min_words.

    Retourne :
    - accepted_df : articles conservés ;
    - rejected_df : articles rejetés.
    """
    if article_column not in dataframe.columns:
        raise KeyError(
            f"La colonne '{article_column}' est absente. "
            f"Colonnes disponibles : {list(dataframe.columns)}"
        )

    if min_words < 1:
        raise ValueError(
            "min_words doit être supérieur ou égal à 1."
        )

    working_df = dataframe.copy()

    # Conservation de la position d'origine.
    working_df["original_index"] = working_df.index

    working_df[article_column] = (
        working_df[article_column]
        .fillna("")
        .astype(str)
        .apply(clean_single_article)
    )

    working_df["word_count_after_cleaning"] = (
        working_df[article_column].apply(count_words)
    )

    short_article_mask = (
        working_df["word_count_after_cleaning"] < min_words
    )

    rejected_df = working_df.loc[short_article_mask].copy()
    rejected_df["rejection_reason"] = (
        "article_below_minimum_word_count"
    )

    accepted_df = working_df.loc[~short_article_mask].copy()

    accepted_df.reset_index(drop=True, inplace=True)
    rejected_df.reset_index(drop=True, inplace=True)

    return accepted_df, rejected_df