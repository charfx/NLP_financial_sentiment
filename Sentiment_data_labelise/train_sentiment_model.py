from pathlib import Path
import numpy as np
import torch
from torch import nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from Model_classification_sentiment.Financial_sentiment_classifier import (
    FinancialSentimentClassifier,
)

from Sentiment_data_labelise.sentiment_dataset import (
    build_dataloaders,
)

TOKENIZER_PATH = Path(
    "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
)

BEST_MODEL_PATH = Path(
    "artifacts/best_c7_sentiment.pt"
)

LABEL_NAMES = [
    "Negative",
    "Neutral",
    "Positive",
]


def train_model(
    num_epochs: int = 5,
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

    print(f"Device : {device}")

    if device.type == "cuda":
        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    # ======================================================
    # DATA
    # ======================================================

    (
        train_loader,
        validation_loader,
        test_loader,
        class_weights,
    ) = build_dataloaders(
        batch_size=batch_size
    )

    print(
        f"Train batches : {len(train_loader)} | "
        f"Validation : {len(validation_loader)} | "
        f"Test : {len(test_loader)}"
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = FinancialSentimentClassifier(
        tokenizer_path=str(TOKENIZER_PATH),
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

    print(
        "Model device :",
        next(model.parameters()).device,
    )

    # ======================================================
    # LOSS + OPTIMIZER
    # ======================================================

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # ======================================================
    # EVALUATION FUNCTION
    # ======================================================

    def evaluate_model(
        model,
        data_loader,
    ):
        model.eval()

        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():

            for batch in data_loader:

                input_ids = batch["input_ids"].to(
                    device,
                    non_blocking=True,
                )

                attention_mask = batch[
                    "attention_mask"
                ].to(
                    device,
                    non_blocking=True,
                )

                labels = batch["label"].to(
                    device,
                    non_blocking=True,
                )

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = criterion(
                    logits,
                    labels,
                )

                total_loss += loss.item()

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                all_labels.extend(
                    labels.cpu().tolist()
                )

                all_predictions.extend(
                    predictions.cpu().tolist()
                )

        return {
            "loss": (
                total_loss
                / len(data_loader)
            ),

            "accuracy": accuracy_score(
                all_labels,
                all_predictions,
            ),

            "precision": precision_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            ),

            "recall": recall_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            ),

            "f1": f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            ),

            "labels": all_labels,
            "predictions": all_predictions,
        }

    # ======================================================
    # TRAINING
    # ======================================================

    best_validation_f1 = -1.0

    BEST_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        # --------------------------------------------------
        # TRAINING BATCHES
        # --------------------------------------------------

        for batch in train_loader:

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_examples += labels.size(0)

        # --------------------------------------------------
        # TRAIN METRICS
        # --------------------------------------------------

        train_loss = (
            total_loss
            / len(train_loader)
        )

        train_accuracy = (
            correct_predictions
            / total_examples
        )

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
        )

        # --------------------------------------------------
        # SIMPLE EPOCH DISPLAY
        # --------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Loss {validation_metrics['loss']:.4f} | "
            f"Val Acc {validation_metrics['accuracy']:.4f} | "
            f"Val F1 {validation_metrics['f1']:.4f}"
        )

        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if (
            validation_metrics["f1"]
            > best_validation_f1
        ):

            best_validation_f1 = (
                validation_metrics["f1"]
            )

            torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),

                "validation_f1": validation_metrics["f1"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],

                "embedding_dim": 256,
                "num_heads": 8,
                "feed_forward_dim": 1024,
                "num_layers": 4,
                "num_classes": 3,
                "max_length": 512,
            },
            BEST_MODEL_PATH,
        )

    # ======================================================
    # FINAL TEST
    # ======================================================

    checkpoint=torch.load(BEST_MODEL_PATH,map_location=device,weights_only=False,)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
    )

    # ======================================================
    # FINAL METRICS
    # ======================================================

    print("\n=== FINAL TEST RESULTS ===")

    print(
        f"Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Macro F1  : "
        f"{test_metrics['f1']:.4f}"
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    print("=== CONFUSION MATRIX ===")

    print(
        confusion_matrix(
            test_metrics["labels"],
            test_metrics["predictions"],
        )
    )

    return model


if __name__ == "__main__":

    model = train_model(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-4,
    )