from pathlib import Path
from turtle import pd
import pandas as pd
import torch

from Embedding.C7_embedding import C7Embedding


def main() -> None:
    tokenizer_path = Path(
        "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
    )

    embedding_model = C7Embedding(
        tokenizer_path=tokenizer_path,
        embedding_dim=256,
        max_length=512,
        dropout=0.1,
    )

    corupus_df = pd.read_parquet("data/processed/articles_cleaned_final.parquet",

                columns=["Headline","Article"],
                )
    headline = corupus_df["Headline"].iloc[1]
    article = corupus_df["Article"].iloc[1]


    input_ids, attention_mask = (
        embedding_model.encode_text(
            headline=headline,
            article=article,
        )
    )

    embeddings = embedding_model(
        input_ids
    )

    print("Input IDs shape :")
    print(input_ids.shape)

    print("\nAttention mask shape :")
    print(attention_mask.shape)

    print("\nDense embeddings shape :")
    print(embeddings.shape)

    print("\nPremier vecteur [CLS] :")
    print(embeddings[0, 0])

    print("\nVecteur d'une position PAD :")

    first_padding_position = int(
        attention_mask[0].sum().item()
    )

    print(
        embeddings[
            0,
            first_padding_position,
        ]
    )

    print("\nNombre de tokens réels :")
    print(
        int(
            attention_mask[0].sum().item()
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    main()