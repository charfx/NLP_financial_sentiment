from pathlib import Path

import pandas as pd
import torch

from Embedding.C7_embedding import C7Embedding
from Transformer.Encoder_block import TransformerEncoderblock


def main() -> None:
    tokenizer_path = Path(
        "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
    )

    corpus_path = Path(
        "data/processed/articles_cleaned_final.parquet"
    )

    corpus_df = pd.read_parquet(
        corpus_path,
        columns=["Headline", "Article"],
    )

    headline = corpus_df.iloc[0]["Headline"]
    article = corpus_df.iloc[0]["Article"]

    embedding_model = C7Embedding(
        tokenizer_path=tokenizer_path,
        embedding_dim=256,
        max_length=512,
        dropout=0.1,
    )

    encoder_block = TransformerEncoderblock(
        embedding_dim=256,
        num_heads=8,
        feed_forward_dim=1024,
        dropout=0.1,
    )

    input_ids, attention_mask = embedding_model.encode_text(
        headline=headline,
        article=article,
    )

    dense_embeddings = embedding_model(
        input_ids
    )

    encoder_output = encoder_block(
        x=dense_embeddings,
        attention_mask=attention_mask,
    )

    print("Input IDs shape :")
    print(input_ids.shape)

    print("\nEmbedding shape :")
    print(dense_embeddings.shape)

    print("\nEncoder block output shape :")
    print(encoder_output.shape)

    print("\nLa forme est conservée :")
    print(
        dense_embeddings.shape
        == encoder_output.shape
    )

    print("\nVecteur CLS contextualisé :")
    print(encoder_output[0, 0])


if __name__ == "__main__":
    torch.manual_seed(42)
    main()