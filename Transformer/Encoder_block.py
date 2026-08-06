## l'encodeur de text du transformers
## compose de multihead attention assurer echange d'information entr tokens 
## un feed forward network integrer non linearite dans chaque token
## residual addition assurer :conservation de l'information precedente
## normalization layers mise a l'echelle de la couche pour stabiliser l'apprentissage
## some dropout layers annuller certaine neurons pour eviter le surapprentissage
##une chose que multiheadattetion attends key_padding_mask logique inverse de attention mask
## attentions mask =[1,1,0] key_padding_mask=[false,false,true]

import torch
from torch import nn


class TransformerEncoderblock(nn.Module):
    """
    Bloc Transformer Encoder complet.

    Architecture :
        Multi-Head Self-Attention
        → Residual Connection
        → Layer Normalization
        → Feed-Forward Network
        → Residual Connection
        → Layer Normalization

    Entrées :
        x :
            [batch_size, sequence_length, embedding_dim]

        attention_mask :
            [batch_size, sequence_length]

            1 = token réel
            0 = token PAD

    Sortie :
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        feed_forward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ======================================================
        # VALIDATION DES PARAMÈTRES
        # ======================================================

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim doit être strictement positif."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads doit être strictement positif."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim doit être divisible par num_heads."
            )

        if feed_forward_dim <= 0:
            raise ValueError(
                "feed_forward_dim doit être strictement positif."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout doit être compris entre 0 et 1."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        # ======================================================
        # MULTI-HEAD SELF-ATTENTION
        # ======================================================

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ======================================================
        # FEED-FORWARD NETWORK
        # ======================================================

        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=feed_forward_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=feed_forward_dim,
                out_features=embedding_dim,
            ),
        )

        # ======================================================
        # LAYER NORMALIZATION
        # ======================================================

        self.norm_after_attention = nn.LayerNorm(
            normalized_shape=embedding_dim,
        )

        self.norm_after_feed_forward = nn.LayerNorm(
            normalized_shape=embedding_dim,
        )

        # ======================================================
        # DROPOUTS DES CONNEXIONS RÉSIDUELLES
        # ======================================================

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Applique :

        1. Multi-Head Self-Attention
        2. Residual Addition + LayerNorm
        3. Feed-Forward Network
        4. Residual Addition + LayerNorm
        """

        if x.ndim != 3:
            raise ValueError(
                "x doit avoir la forme "
                "[batch_size, sequence_length, embedding_dim]."
            )

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"La dernière dimension de x doit être "
                f"{self.embedding_dim}, reçue : {x.size(-1)}."
            )

        key_padding_mask = None

        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask doit avoir la forme "
                    "[batch_size, sequence_length]."
                )

            if attention_mask.shape != x.shape[:2]:
                raise ValueError(
                    "attention_mask doit correspondre aux dimensions "
                    "[batch_size, sequence_length] de x."
                )

            # Tokenizer :
            # 1 = token réel
            # 0 = PAD
            #
            # MultiheadAttention :
            # False = position utilisable
            # True  = position à ignorer
            key_padding_mask = attention_mask == 0

        # ======================================================
        # 1. MULTI-HEAD SELF-ATTENTION
        # ======================================================

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual connection + dropout + LayerNorm
        x = self.norm_after_attention(
            x
            + self.attention_dropout(
                attention_output
            )
        )

        # ======================================================
        # 2. FEED-FORWARD NETWORK
        # ======================================================

        feed_forward_output = self.feed_forward(x)

        # Deuxième residual connection + dropout + LayerNorm
        x = self.norm_after_feed_forward(
            x
            + self.feed_forward_dropout(
                feed_forward_output
            )
        )

        return x