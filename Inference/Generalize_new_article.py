## nous allons generalizer sur un nouveau article
## on appellant tout le process d'un NLP 
##Tokenization - embedding - transformer - model -.pt entrainement- .classify


from pathlib import Path
import httpx 
import torch
from Model_classification_sentiment.Financial_sentiment_classifier import (
    FinancialSentimentClassifier,
)

TOKENIZER_PATH = Path(
    "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
)

CHECKPOINT_PATH = Path(
    "artifacts/best_c7_sentiment.pt"
)

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}
NEWS_API_URL=(  "https://newsdata.io/api/1/latest"
    "?apikey=pub_df9ab49634dd45c8a8b29099091a88fb"
    "&q=finance")

def fetch_latest_article() -> tuple[str, str]:
    """
    Recupere le dernier article financier disponible
    via l'API NewsData.io (title + description).
    """

    response = httpx.get(NEWS_API_URL)
    response.raise_for_status()

    data = response.json()
    articles = data.get("results", [])

    for item in articles:

        title = item.get("title")
        description = item.get("description")

        if title and description:
            return title, description

    raise ValueError(
        "Aucun article avec title et description "
        "trouve dans la reponse de l'API."
    )
def load_model(device: torch.device) -> FinancialSentimentClassifier:
    """
    Reconstruit l'architecture puis recharge
    les poids appris depuis le checkpoint .pt.
    """

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = FinancialSentimentClassifier(
        tokenizer_path=str(TOKENIZER_PATH),
        embedding_dim=checkpoint["embedding_dim"],
        max_length=checkpoint["max_length"],
        num_heads=checkpoint["num_heads"],
        feed_forward_dim=checkpoint["feed_forward_dim"],
        num_layers=checkpoint["num_layers"],
        classifier_hidden_dim=128,
        num_classes=checkpoint["num_classes"],
        dropout=0.1,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


def classify_article(
    model: FinancialSentimentClassifier,
    headline: str,
    article: str,
    device: torch.device,
):
    """
    Classifie un nouvel article financier.
    """

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    input_ids, attention_mask = (
        model.embedding.encode_text(
            headline=headline,
            article=article,
        )
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .tolist()
    )

    return {
        "prediction": LABELS[predicted_class],
        "negative_probability": probabilities[0],
        "neutral_probability": probabilities[1],
        "positive_probability": probabilities[2],
    }


def main() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    model = load_model(device)

    # --------------------------------------------------
    # NEW ARTICLE
    # --------------------------------------------------

    headline,article=fetch_latest_article()
    print("\n headline recuppere:",headline)
    print("\n description recupere",article)
    result = classify_article(
        model=model,
        headline=headline,
        article=article,
        device=device,
    )

    print("\n=== SENTIMENT CLASSIFICATION ===")

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Negative : "
        f"{result['negative_probability']:.4f}"
    )

    print(
        f"Neutral  : "
        f"{result['neutral_probability']:.4f}"
    )

    print(
        f"Positive : "
        f"{result['positive_probability']:.4f}"
    )


if __name__ == "__main__":
    main()