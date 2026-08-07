from pathlib import Path

import torch
from torch import nn

from Model_classification_sentiment.Financial_sentiment_classifier import (
    FinancialSentimentClassifier,
)

from Sentiment_data_labelise.sentiment_dataset import (
    build_dataloaders,
)


TOKENIZER_PATH = Path(
    "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
)


def train_model(
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
) -> FinancialSentimentClassifier:

    # ======================================================
    # DEVICE
    # ======================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    # ======================================================
    # DATA
    # ======================================================

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = build_dataloaders(
        batch_size=batch_size
    )

    print(
        "Train batches :",
        len(train_loader)
    )

    print(
        "Validation batches :",
        len(validation_loader)
    )

    print(
        "Test batches :",
        len(test_loader)
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = FinancialSentimentClassifier(
        tokenizer_path=str(
            TOKENIZER_PATH
        ),
        embedding_dim=256,
        max_length=512,
        num_heads=8,
        feed_forward_dim=1024,
        num_layers=4,
        classifier_hidden_dim=128,
        num_classes=3,
        dropout=0.1,
    )

    model = model.to(device)

    # ======================================================
    # LOSS
    # ======================================================

    criterion = nn.CrossEntropyLoss()

    # ======================================================
    # OPTIMIZER
    # ======================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for epoch in range(num_epochs):

        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("=" * 70)

        model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        for batch_idx, batch in enumerate(train_loader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # ------------------------------------------
            # Reset gradients
            # ------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # ------------------------------------------
            # Loss
            # ------------------------------------------

            loss = criterion(
                logits,
                labels,
            )

            # ------------------------------------------
            # Backpropagation
            # ------------------------------------------

            loss.backward()

            # ------------------------------------------
            # Update parameters
            # ------------------------------------------

            optimizer.step()

            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_examples += labels.size(0)

            # ======================================================
            # Affichage toutes les 20 batches
            # ======================================================

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:

                progress = (
                    (batch_idx + 1)
                    / len(train_loader)
                ) * 100

                current_accuracy = (
                    correct_predictions
                    / total_examples
                )

                print(
                    f"[{progress:6.2f}%] "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss = {loss.item():.4f} | "
                    f"Accuracy = {current_accuracy:.4f}"
                )
            average_loss = (
                total_loss
                / len(train_loader)
            )

            accuracy = (
                correct_predictions
                / total_examples
            )

        

    return model

if __name__ == "__main__":

    model = train_model(
        num_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
    )