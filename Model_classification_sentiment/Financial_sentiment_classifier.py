## le model qui va appeller les tokenizer
##embedding 
##transformer blocks 
## usage du [CLS]
##livrer la classificatin finale base sur un reseau neuronal complet
##qui etudie les logit et applique softmax et livre un sentiment probabiliste

import torch
import torch.nn as nn

from Embedding.C7_embedding import C7Embedding
from Transformer.Transformer_encoder import TransformerEncoder

class FinancialSentimentClassifier(nn.Module):

    def __init__(
        self,
        tokenizer_path: str,
        embedding_dim: int = 256,
        max_length: int = 512,
        num_heads: int = 8,
        feed_forward_dim: int = 1024,
        num_layers: int = 4,
        classifier_hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = C7Embedding(
            tokenizer_path=tokenizer_path,
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout,
        )

        self.transformer = TransformerEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(
                embedding_dim,
                classifier_hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                classifier_hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # 1. Token + positional embeddings
        embeddings = self.embedding(
            input_ids
        )

        # 2. Contextualisation Transformer
        contextualized_output = self.transformer(
            x=embeddings,
            attention_mask=attention_mask,
        )

        # 3. Représentation globale de l'article
        cls_vector = contextualized_output[:, 0, :]

        # 4. Classification
        logits = self.classifier(
            cls_vector
        )

        return logits