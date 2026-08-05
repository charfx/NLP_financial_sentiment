from pathlib import Path

import torch
from tokenizers import Tokenizer


TOKENIZER_PATH = Path(
    "artifacts/tokenizer/c7_wordpiece_v1/tokenizer.json"
)

MAX_LENGTH = 512


def load_c7_tokenizer() -> Tokenizer:
    """
    Charge le tokenizer WordPiece entraîné.
    """
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_PATH.resolve()}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    pad_token_id = tokenizer.token_to_id("[PAD]")

    if pad_token_id is None:
        raise ValueError(
            "Le token [PAD] est absent du vocabulaire."
        )

    tokenizer.enable_truncation(
        max_length=MAX_LENGTH,
    )

    tokenizer.enable_padding(
        direction="right",
        pad_id=pad_token_id,
        pad_token="[PAD]",
        length=MAX_LENGTH,
    )

    return tokenizer


def encode_article(
    tokenizer: Tokenizer,
    headline: str,
    article: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transforme un article en :
    - input_ids ;
    - attention_mask.

    Les tenseurs retournés ont une forme :
    [1, MAX_LENGTH]
    """

    text = (
        f"[TITLE] {headline} "
        f"[ARTICLE] {article}"
    )

    encoding = tokenizer.encode(text)

    input_ids = torch.tensor(
        encoding.ids,
        dtype=torch.long,
    ).unsqueeze(0)

    attention_mask = torch.tensor(
        encoding.attention_mask,
        dtype=torch.long,
    ).unsqueeze(0)

    return input_ids, attention_mask


def main() -> None:
    tokenizer = load_c7_tokenizer()

    headline = (
        "NVIDIA Raises Revenue Forecast"
    )

    article = (
        "NVIDIA reported stronger demand for artificial "
        "intelligence chips and increased its revenue outlook."
    )

    input_ids, attention_mask = encode_article(
        tokenizer=tokenizer,
        headline=headline,
        article=article,
    )

    encoding = tokenizer.encode(
        f"[TITLE] {headline} [ARTICLE] {article}"
    )

    print("Tokens :")
    print(encoding.tokens[:30])

    print("\nToken IDs :")
    print(input_ids[0, :30])

    print("\nAttention mask :")
    print(attention_mask[0, :30])

    print("\nInput IDs shape :")
    print(input_ids.shape)

    print("\nAttention mask shape :")
    print(attention_mask.shape)


if __name__ == "__main__":
    main()