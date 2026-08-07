from pathlib import Path

import pandas as pd
import torch

from Embedding.C7_embedding import C7Embedding
from Transformer.Transformer_encoder import TransformerEncoder


def main() -> None:

    # ======================================================
    # PATHS
    # ======================================================

    tokenizer_path = Path(
        "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
    )

    corpus_path = Path(
        "data/processed/articles_cleaned_final.parquet"
    )

    # ======================================================
    # LOAD REAL ARTICLE
    # ======================================================

    corpus_df = pd.read_parquet(
        corpus_path,
        columns=["Headline", "Article"],
    )

    headline = corpus_df.iloc[0]["Headline"]
    article = corpus_df.iloc[0]["Article"]

    print("Headline :")
    print(headline)

    # ======================================================
    # EMBEDDING MODEL
    # ======================================================

    embedding_model = C7Embedding(
        tokenizer_path=tokenizer_path,
        embedding_dim=256,
        max_length=512,
        dropout=0.1,
    )

    # ======================================================
    # TRANSFORMER ENCODER
    # ======================================================

    transformer_encoder = TransformerEncoder(
        embedding_dim=256,
        num_heads=8,
        feed_forward_dim=1024,
        num_layers=4,
        dropout=0.1,
    )

    # ======================================================
    # TOKENIZATION
    # ======================================================

    input_ids, attention_mask = (
        embedding_model.encode_text(
            headline=headline,
            article=article,
        )
    )

    # ======================================================
    # EMBEDDING
    # ======================================================

    dense_embeddings = embedding_model(
        input_ids
    )

    # ======================================================
    # TRANSFORMER
    # ======================================================

    contextualized_output = transformer_encoder(
        x=dense_embeddings,
        attention_mask=attention_mask,
    )

    # ======================================================
    # CLS REPRESENTATION
    # ======================================================

    cls_vector = contextualized_output[:, 0, :]

    # ======================================================
    # RESULTS
    # ======================================================

    print("\nInput IDs shape :")
    print(input_ids.shape)

    print("\nAttention Mask shape :")
    print(attention_mask.shape)

    print("\nEmbedding output shape :")
    print(dense_embeddings.shape)

    print("\nTransformer output shape :")
    print(contextualized_output.shape)

    print("\nCLS vector shape :")
    print(cls_vector.shape)

    print("\nPremier vecteur CLS contextualisé :")
    print(cls_vector[0])


if __name__ == "__main__":
    torch.manual_seed(42)
    main()