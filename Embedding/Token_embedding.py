## token embedding ( by the way il seront rassembler dans un seul file avec positionnal embedding)
import torch
from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,padding_idx: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(
                "vocab_size doit être strictement positif."
            )

        if embed_dim <= 0:
            raise ValueError(
                "embed_dim doit être strictement positif."
            )

        if not 0 <= padding_idx < vocab_size:
            raise ValueError(
                "padding_idx doit être compris dans le vocabulaire."
            )
        ##Most important line of the code, it will create a embedding layer with the vocab size and the embedding dimension, and the padding index will be used to ignore the padding tokens during training.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cherche le vecteur correspondant à chaque token ID.
        """

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids doit avoir le dtype torch.long."
            )

        return self.embedding(input_ids)
    