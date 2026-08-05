import torch

from Embedding.Token_embedding import TokenEmbedding


def main() -> None:
    vocab_size = 30_000
    embedding_dim = 256
    padding_idx = 0

    token_embedding = TokenEmbedding(
        vocab_size=vocab_size,
        embed_dim=embedding_dim,
        padding_idx=padding_idx,
    )

    input_ids = torch.tensor(
        [
            [
                2,
                5,
                4595,
                11313,
                4231,
                3,
                0,
                0,
            ]
        ],
        dtype=torch.long,
    )

    embeddings = token_embedding(input_ids)

    print("Input IDs shape :")
    print(input_ids.shape)

    print("\nEmbeddings shape :")
    print(embeddings.shape)

    print("\nVecteur du token ID 4595 :")
    print(embeddings[0, 2])

    print("\nVecteur du PAD :")
    print(embeddings[0, 6])

    print("\nPAD entièrement nul :")
    print(
        torch.all(
            embeddings[0, 6] == 0
        ).item()
    )


if __name__ == "__main__":
    main()