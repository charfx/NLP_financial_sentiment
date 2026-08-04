## le noeud principal de l'entrainement du tokenizer Wordpiece est de construire le vocabulaire a partir du corpus train

## ou on appelle les deux element construction + tokenization

from pathlib import Path

import pandas as pd

from Tokenization.Construction_txt import article_text_iterator
from Tokenization.Train_tokenization import train_c7_tokenizer


def main() -> None:
    train_corpus_path = Path(
        "data/splits/tokenizer_train.parquet"
    )

    output_directory = Path(
        "artifacts/tokenizer/c7_wordpiece_v1"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_size = len(
        pd.read_parquet(
            train_corpus_path,
            columns=["Headline"],
        )
    )

    corpus_iterator = article_text_iterator(
        train_corpus_path
    )

    tokenizer = train_c7_tokenizer(
        text_iterator=corpus_iterator,
        corpus_size=corpus_size,
    )

    output_path = output_directory / "tokenizer.json"

    tokenizer.save(str(output_path))

    print(f"Tokenizer sauvegardé : {output_path}")
    print(f"Taille réelle du vocabulaire : {tokenizer.get_vocab_size():,}")


if __name__ == "__main__":
    main()