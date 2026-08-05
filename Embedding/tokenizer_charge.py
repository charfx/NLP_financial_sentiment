## chargement du tokenizer.json 

from pathlib import Path
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("artifacts/tokenizer/c7_wordpiece_V1/tokenizer.json")


def main()  -> None:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")

    
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    vocab_size=tokenizer.get_vocab_size()
    pad_token_id=tokenizer.token_to_id("[PAD]")
    cls_token_id=tokenizer.token_to_id("[CLS]")
    sep_token_id=tokenizer.token_to_id("[SEP]")

    print(f"vocabulary size: {vocab_size}")
    print(f"pad token id: {pad_token_id}")
    print(f"cls token id: {cls_token_id}")
    print(f"sep token id: {sep_token_id}")

if __name__ == "__main__":
    main()