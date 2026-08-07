## le model noyau de toute l'architecture qui va appeller
## encoder block 4 fois chaque fois avec different paramtre partiellement
## chaque block va renforcer le context jusqu'a atteindre la meuilleur
## representation contextuelle

import torch
from torch import nn

from Transformer.Encoder_block import TransformerEncoderblock


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder composé de plusieurs Encoder Blocks.

    Entrée :
        x :
        [batch_size, sequence_length, embedding_dim]

        attention_mask :
        [batch_size, sequence_length]

    Sortie :
        contextualized representations
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        feed_forward_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(
                "num_layers doit être strictement positif."
            )

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                TransformerEncoderblock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            normalized_shape=embedding_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "x doit avoir la forme "
                "[batch_size, sequence_length, embedding_dim]."
            )

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"embedding_dim attendu : {self.embedding_dim}, "
                f"reçu : {x.size(-1)}."
            )

        for layer in self.layers:
            x = layer(
                x=x,
                attention_mask=attention_mask,
            )

        x = self.final_norm(x)

        return x