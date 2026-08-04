import os

from datasets import load_dataset

from Filter_data.first_cleaning import apply_first_cleaning
from Filter_data.second_cleaning import apply_second_cleaning


def main() -> None:
    print("Chargement du dataset Bloomberg...")

    dataset = load_dataset(
        "danidanou/Bloomberg_Financial_News",
        split="train",
    )

    train_df = dataset.to_pandas()

    print(f"Articles initiaux : {len(train_df):,}")

    # ==========================================================
    # FIRST CLEANING
    # ==========================================================

    first_accepted_df, first_rejected_df = apply_first_cleaning(
        dataframe=train_df,
        article_column="Article",
        min_words=20,
    )

    print("\n=== FIRST CLEANING REPORT ===")

    print(
        f"Articles conservés après first cleaning : "
        f"{len(first_accepted_df):,}"
    )

    print(
        f"Articles rejetés pour faible longueur   : "
        f"{len(first_rejected_df):,}"
    )

    # ==========================================================
    # SECOND CLEANING
    # ==========================================================

    print("\nAnalyse et suppression des documents tabulaires...")

    second_accepted_df, second_rejected_df = apply_second_cleaning(
        dataframe=first_accepted_df,
        article_column="Article",
        max_narrative_density=0.45,
        min_long_separators=3,
        min_numeric_groups=80,
        min_numeric_segment_ratio=0.15,
        min_table_like_segment_ratio=0.20,
        min_average_words_per_sentence=35.0,
    )

    print("\n=== SECOND CLEANING REPORT ===")

    print(
        f"Articles reçus après first cleaning     : "
        f"{len(first_accepted_df):,}"
    )

    print(
        f"Documents tabulaires rejetés            : "
        f"{len(second_rejected_df):,}"
    )

    print(
        f"Articles conservés après second cleaning: "
        f"{len(second_accepted_df):,}"
    )

    # ==========================================================
    # GLOBAL REPORT
    # ==========================================================

    total_rejected = (
        len(first_rejected_df)
        + len(second_rejected_df)
    )

    print("\n=== GLOBAL CLEANING SUMMARY ===")

    print(
        f"Articles initiaux                     : "
        f"{len(train_df):,}"
    )

    print(
        f"Articles courts rejetés               : "
        f"{len(first_rejected_df):,}"
    )

    print(
        f"Documents tabulaires rejetés          : "
        f"{len(second_rejected_df):,}"
    )

    print(
        f"Total articles rejetés                : "
        f"{total_rejected:,}"
    )

    print(
        f"Corpus final conservé                 : "
        f"{len(second_accepted_df):,}"
    )

    print(
        f"Pourcentage conservé                  : "
        f"{len(second_accepted_df) / len(train_df) * 100:.2f}%"
    )

    # ==========================================================
    # USDA CHECK
    # ==========================================================

    headline_to_find = (
        "USDA Boxed Beef Cutout Closing Prices for October 6"
    )

    usda_in_accepted = second_accepted_df[
        second_accepted_df["Headline"].eq(headline_to_find)
    ]

    usda_in_rejected = second_rejected_df[
        second_rejected_df["Headline"].eq(headline_to_find)
    ]

    print("\n=== USDA FINAL CHECK ===")

    if not usda_in_rejected.empty:
        print("USDA a bien été rejeté par le second cleaning.")

        columns = [
            "Headline",
            "narrative_density",
            "long_separator_count",
            "numeric_group_count",
            "numeric_segment_ratio",
            "table_like_segment_ratio",
            "average_words_per_sentence",
            "rejection_reason",
        ]

        print(
            usda_in_rejected[columns]
            .iloc[0]
            .to_string()
        )

    elif not usda_in_accepted.empty:
        print(
            "Attention : USDA est encore présent "
            "dans le corpus final."
        )

    else:
        print("Article USDA introuvable.")

    # ==========================================================
    # EXPORT
    # ==========================================================

    output_directory = "data/processed"
    os.makedirs(output_directory, exist_ok=True)

    final_dataset_path = os.path.join(
        output_directory,
        "articles_cleaned_final.parquet",
    )

    first_rejected_path = os.path.join(
        output_directory,
        "articles_rejected_short.parquet",
    )

    second_rejected_path = os.path.join(
        output_directory,
        "articles_rejected_tabular.parquet",
    )

    second_accepted_df.to_parquet(
        final_dataset_path,
        index=False,
    )

    first_rejected_df.to_parquet(
        first_rejected_path,
        index=False,
    )

    second_rejected_df.to_parquet(
        second_rejected_path,
        index=False,
    )

    print("\n=== EXPORT COMPLETED ===")

    print(f"Corpus final           : {final_dataset_path}")
    print(f"Articles courts rejetés: {first_rejected_path}")
    print(f"Tableaux rejetés       : {second_rejected_path}")


if __name__ == "__main__":
    main()