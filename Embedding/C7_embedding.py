## ici le vrai fichier embedding quiva faire a la fois positionnal et token 
## embedding et livre un vecteur finale dense 
## cela sera appliquer dans l'entrainnement dans un fichier test_final_embeding.py
## ou l'entrainement sera fait batch par batch et le vecteur final sera calculer pour chaque batch

from pathlib import Path
import torch
from torch import nn
from tokenizers import Tokenizer


class C7Embedding(nn.Module):
    """
    Pipeline complet :

    texte financier
    → tokenizer
    → input_ids
    → token embedding
    → positional embedding
    → somme
    → dropout

    Sortie :
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        embedding_dim: int = 256,
        max_length: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer introuvable : {tokenizer_path.resolve()}"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim doit être strictement positif."
            )

        if max_length <= 0:
            raise ValueError(
                "max_length doit être strictement positif."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout doit être compris entre 0 et 1."
            )

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.max_length = max_length
        self.embedding_dim = embedding_dim

        vocab_size = self.tokenizer.get_vocab_size()
        pad_token_id = self.tokenizer.token_to_id("[PAD]")

        if pad_token_id is None:
            raise ValueError(
                "Le token [PAD] est absent du tokenizer."
            )

        self.tokenizer.enable_truncation(
            max_length=max_length,
        )

        self.tokenizer.enable_padding(
            direction="right",
            pad_id=pad_token_id,
            pad_token="[PAD]",
            length=max_length,
        )

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(
            p=dropout,
        )

    def encode_text(
        self,
        headline: str,
        article: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transforme un article en input_ids et attention_mask.
        """

        text = (
            f"[TITLE] {headline} "
            f"[ARTICLE] {article}"
        )

        encoding = self.tokenizer.encode(text)

        input_ids = torch.tensor(
            encoding.ids,
            dtype=torch.long,
        ).unsqueeze(0)

        attention_mask = torch.tensor(
            encoding.attention_mask,
            dtype=torch.long,
        ).unsqueeze(0)

        return input_ids, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transforme les IDs en représentation dense complète.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids doit avoir la forme "
                "[batch_size, sequence_length]."
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids doit être de type torch.long."
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"La séquence contient {sequence_length} positions, "
                f"mais max_length vaut {self.max_length}."
            )

        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=input_ids.device,
        )

        position_ids = position_ids.unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        token_vectors = self.token_embedding(
            input_ids
        )

        position_vectors = self.position_embedding(
            position_ids
        )

        embeddings = (
            token_vectors
            + position_vectors
        )

        embeddings = self.dropout(
            embeddings
        )

        return embeddings