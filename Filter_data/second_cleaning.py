import re
from typing import Dict, List, Tuple

import pandas as pd


# ==========================================================
# REGEX PATTERNS
# ==========================================================

# Séparation approximative des phrases.
SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)

# Séparateurs typiques des tableaux :
# ----------------
# ================
LONG_SEPARATOR_PATTERN = re.compile(
    r"[-_=*]{8,}"
)

# Détection des groupes numériques :
# 184.82
# 4,769,758
# 10/05
# 600-900
# 12.5%
NUMERIC_GROUP_PATTERN = re.compile(
    r"""
    (?<!\w)
    [+-]?
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
        |
        \d+/\d+
        |
        \d+-\d+
    )
    %?
    (?!\w)
    """,
    flags=re.VERBOSE,
)

# Liste simple de verbes fréquents dans les articles financiers.
COMMON_VERB_PATTERN = re.compile(
    r"\b("
    r"is|are|was|were|be|been|being|"
    r"has|have|had|"
    r"do|does|did|"
    r"will|would|could|should|may|might|must|can|"
    r"said|says|reported|reports|announced|"
    r"expects|expected|forecast|forecasts|"
    r"rose|rise|rises|fell|fall|falls|"
    r"declined|declines|increased|increases|"
    r"grew|grow|grows|cut|cuts|"
    r"raised|raises|reduced|reduces|"
    r"maintained|maintains|"
    r"traded|trades|closed|opened|"
    r"plans|planned|agreed"
    r")\b",
    flags=re.IGNORECASE,
)


# ==========================================================
# BASIC UTILITIES
# ==========================================================

def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """
    Effectue une division sans risque de division par zéro.
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def count_words(text: str) -> int:
    """
    Compte approximativement les mots.
    """
    if not isinstance(text, str):
        return 0

    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """
    Sépare approximativement un texte en phrases.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    return [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if sentence.strip()
    ]


def split_structural_segments(text: str) -> List[str]:
    """
    Construit des segments virtuels.

    Le dataset Bloomberg contient souvent les articles
    sous forme d'une seule ligne.

    Nous utilisons donc :
    - les longues suites de séparateurs ;
    - les fins de phrases ;
    comme frontières structurelles.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    segmented_text = LONG_SEPARATOR_PATTERN.sub(
        "\n",
        text,
    )

    segmented_text = SENTENCE_SPLIT_PATTERN.sub(
        "\n",
        segmented_text,
    )

    return [
        segment.strip()
        for segment in segmented_text.splitlines()
        if segment.strip()
    ]


def calculate_alphabetic_ratio(text: str) -> float:
    """
    Calcule la proportion de lettres parmi les caractères
    alphanumériques d'un fragment.

    Les espaces et la ponctuation ne sont pas inclus.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    alphabetic_count = sum(
        character.isalpha()
        for character in text
    )

    digit_count = sum(
        character.isdigit()
        for character in text
    )

    return safe_ratio(
        alphabetic_count,
        alphabetic_count + digit_count,
    )


def count_numeric_groups(text: str) -> int:
    """
    Compte les groupes numériques présents dans un texte.
    """
    if not isinstance(text, str):
        return 0

    return len(
        NUMERIC_GROUP_PATTERN.findall(text)
    )


# ==========================================================
# STRUCTURAL CLASSIFICATION FUNCTIONS
# ==========================================================

def is_narrative_sentence(
    sentence: str,
    min_words: int = 6,
    min_alphabetic_ratio: float = 0.80,
) -> bool:
    """
    Détermine si un fragment ressemble à une phrase narrative.

    Critères :
    - au moins min_words ;
    - majorité claire de lettres ;
    - présence probable d'un verbe ;
    - absence d'un long séparateur.
    """
    if not isinstance(sentence, str):
        return False

    if count_words(sentence) < min_words:
        return False

    if (
        calculate_alphabetic_ratio(sentence)
        < min_alphabetic_ratio
    ):
        return False

    if LONG_SEPARATOR_PATTERN.search(sentence):
        return False

    if not COMMON_VERB_PATTERN.search(sentence):
        return False

    return True


def is_numeric_segment(
    segment: str,
    min_numeric_groups: int = 3,
    max_alphabetic_ratio: float = 0.65,
) -> bool:
    """
    Détecte un segment dominé par des valeurs numériques.

    Exemple :
    10/05 110 90 24 67 291 184.24 170.27
    """
    if not isinstance(segment, str) or not segment:
        return False

    numeric_groups = count_numeric_groups(segment)

    return (
        numeric_groups >= min_numeric_groups
        and calculate_alphabetic_ratio(segment)
        <= max_alphabetic_ratio
    )


def is_short_segment(
    segment: str,
    max_words: int = 5,
) -> bool:
    """
    Détecte un segment très court.

    Ce critère seul ne provoque aucun rejet.
    """
    segment_word_count = count_words(segment)

    return 0 < segment_word_count <= max_words


def is_table_like_segment(segment: str) -> bool:
    """
    Détecte un segment ressemblant à une ligne de tableau.

    Exemples :
    Primal Rib 289.70 231.58

    10/05 110 90 24 67 291
    """
    if not isinstance(segment, str) or not segment:
        return False

    words = segment.split()

    if not words:
        return False

    numeric_groups = count_numeric_groups(segment)

    numeric_group_ratio = safe_ratio(
        numeric_groups,
        len(words),
    )

    if (
        numeric_groups >= 2
        and numeric_group_ratio >= 0.35
    ):
        return True

    if is_numeric_segment(segment):
        return True

    return False


# ==========================================================
# DOCUMENT ANALYSIS
# ==========================================================

def analyze_document_structure(
    text: str,
) -> Dict[str, float]:
    """
    Produit une fiche technique structurelle complète
    pour un article individuel.

    Cette fonction ne supprime rien.
    """
    if not isinstance(text, str):
        text = ""

    sentences = split_sentences(text)
    segments = split_structural_segments(text)

    word_count = count_words(text)
    character_count = len(text)

    sentence_count = len(sentences)
    segment_count = len(segments)

    narrative_sentence_count = sum(
        is_narrative_sentence(sentence)
        for sentence in sentences
    )

    numeric_segment_count = sum(
        is_numeric_segment(segment)
        for segment in segments
    )

    short_segment_count = sum(
        is_short_segment(segment)
        for segment in segments
    )

    table_like_segment_count = sum(
        is_table_like_segment(segment)
        for segment in segments
    )

    long_separator_count = len(
        LONG_SEPARATOR_PATTERN.findall(text)
    )

    numeric_group_count = count_numeric_groups(text)

    narrative_density = safe_ratio(
        narrative_sentence_count,
        sentence_count,
    )

    numeric_segment_ratio = safe_ratio(
        numeric_segment_count,
        segment_count,
    )

    short_segment_ratio = safe_ratio(
        short_segment_count,
        segment_count,
    )

    table_like_segment_ratio = safe_ratio(
        table_like_segment_count,
        segment_count,
    )

    average_words_per_sentence = safe_ratio(
        word_count,
        sentence_count,
    )

    average_words_per_segment = safe_ratio(
        word_count,
        segment_count,
    )

    return {
        "structure_word_count": word_count,
        "structure_character_count": character_count,
        "structure_sentence_count": sentence_count,
        "structure_segment_count": segment_count,
        "narrative_sentence_count": (
            narrative_sentence_count
        ),
        "narrative_density": narrative_density,
        "long_separator_count": long_separator_count,
        "numeric_group_count": numeric_group_count,
        "numeric_segment_count": numeric_segment_count,
        "numeric_segment_ratio": numeric_segment_ratio,
        "short_segment_count": short_segment_count,
        "short_segment_ratio": short_segment_ratio,
        "table_like_segment_count": (
            table_like_segment_count
        ),
        "table_like_segment_ratio": (
            table_like_segment_ratio
        ),
        "average_words_per_sentence": (
            average_words_per_sentence
        ),
        "average_words_per_segment": (
            average_words_per_segment
        ),
    }


def add_structure_features(
    dataframe: pd.DataFrame,
    article_column: str = "Article",
) -> pd.DataFrame:
    """
    Applique analyze_document_structure() à tous les articles
    et ajoute les métriques sous forme de nouvelles colonnes.
    """
    if article_column not in dataframe.columns:
        raise KeyError(
            f"La colonne '{article_column}' est absente. "
            f"Colonnes disponibles : {list(dataframe.columns)}"
        )

    working_df = dataframe.copy()

    articles = (
        working_df[article_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    feature_records = [
        analyze_document_structure(article)
        for article in articles
    ]

    features_df = pd.DataFrame(
        feature_records,
        index=working_df.index,
    )

    return pd.concat(
        [
            working_df,
            features_df,
        ],
        axis=1,
    )


# ==========================================================
# SECOND CLEANING
# ==========================================================

def apply_second_cleaning(
    dataframe: pd.DataFrame,
    article_column: str = "Article",
    max_narrative_density: float = 0.45,
    min_long_separators: int = 3,
    min_numeric_groups: int = 80,
    min_numeric_segment_ratio: float = 0.15,
    min_table_like_segment_ratio: float = 0.20,
    min_average_words_per_sentence: float = 35.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Génère les métriques structurelles puis sépare :

    - accepted_df : articles narratifs conservés ;
    - rejected_df : documents tabulaires rejetés.
    """

    structured_df = add_structure_features(
        dataframe=dataframe,
        article_column=article_column,
    )

    # Règle 1 :
    # Plusieurs séparateurs, beaucoup de nombres
    # et une forte proportion de segments tabulaires.
    strongly_tabular_mask = (
        (
            structured_df["long_separator_count"]
            >= min_long_separators
        )
        &
        (
            structured_df["numeric_group_count"]
            >= min_numeric_groups
        )
        &
        (
            structured_df["table_like_segment_ratio"]
            >= min_table_like_segment_ratio
        )
    )

    # Règle 2 :
    # Faible densité narrative, confirmée par
    # des segments numériques ou tabulaires.
    low_narrative_tabular_mask = (
        (
            structured_df["narrative_density"]
            < max_narrative_density
        )
        &
        (
            (
                structured_df["numeric_segment_ratio"]
                >= min_numeric_segment_ratio
            )
            |
            (
                structured_df["table_like_segment_ratio"]
                >= min_table_like_segment_ratio
            )
        )
        &
        (
            structured_df["numeric_group_count"]
            >= min_numeric_groups
        )
    )

    # Règle 3 :
    # Tableau aplati produisant de très longues
    # pseudo-phrases remplies de données.
    flattened_table_mask = (
        (
            structured_df["average_words_per_sentence"]
            >= min_average_words_per_sentence
        )
        &
        (
            structured_df["numeric_group_count"]
            >= min_numeric_groups
        )
        &
        (
            structured_df["long_separator_count"]
            >= min_long_separators
        )
    )

    rejected_mask = (
        strongly_tabular_mask
        |
        low_narrative_tabular_mask
        |
        flattened_table_mask
    )

    accepted_df = structured_df.loc[
        ~rejected_mask
    ].copy()

    rejected_df = structured_df.loc[
        rejected_mask
    ].copy()

    rejected_df["rejection_reason"] = (
        "predominantly_tabular_or_numeric_document"
    )

    accepted_df.reset_index(
        drop=True,
        inplace=True,
    )

    rejected_df.reset_index(
        drop=True,
        inplace=True,
    )

    return accepted_df, rejected_df