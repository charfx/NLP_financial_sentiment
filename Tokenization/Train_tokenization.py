## Script du tokenizer training qui va recevoir article par article en format txt depuis construction_txt.py
##lecture title + article et consturtion du vocabulaire
## definiton des caractere et mots speciaux pour le tokenizer
## Model utilise Wordpiece cased   
#  
from anyio import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.trainers import WordPieceTrainer
from collections.abc import Iterable, Iterable, Iterator
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[TITLE]",
    "[ARTICLE]",
]

VOCAB_SIZE = 30_000  # Taille du vocabulaire souhaitée pour le tokenizer
MIN_FREQUENCY = 2  # Fréquence minimale pour qu'un mot soit inclus dans le vocabulaire
MAX_LENGTH = 512  # Longueur maximale des séquences tokenisées


def build_empty_wordpiece_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]",
        )
    )

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
        ]
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.BertPreTokenizer()
    )

    return tokenizer
## defintion du trainer pour le tokenizer Wordpiece
def build_wordpiece_trained() -> WordPieceTrainer:
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        continuing_subword_prefix="##",
    )
    return  trainer
##recevoir l'iterator du text car le constructeur du txt genere text par text facon consecutive pour articels 

def train_c7_tokenizer(
    text_iterator: Iterable[str],
    corpus_size: int,
) -> Tokenizer:
    """
    Entraîne le tokenizer WordPiece sur le corpus train.
    """
    tokenizer = build_empty_wordpiece_tokenizer()
    trainer = build_wordpiece_trained()

    tokenizer.train_from_iterator(
        iterator=text_iterator,
        trainer=trainer,
        length=corpus_size,
    )
    configure_post_processor(tokenizer)

    return tokenizer

## l'ajout de certain element vecteur speciaux CSL et SEP ... 

def configure_post_processor(
    tokenizer: Tokenizer,
) -> None:
    """
    Ajoute automatiquement [CLS] au début
    et [SEP] à la fin de chaque séquence.
    """
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    if cls_id is None or sep_id is None:
        raise ValueError(
            "[CLS] ou [SEP] absent du vocabulaire."
        )

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )
## truncation a comme role decoupage du txt si il depasse la longueur max de 512 tokens
def configure_truncation(
    tokenizer: Tokenizer,
    Max_length: int = MAX_LENGTH,
) -> None:
    
    tokenizer.enable_truncation(
        max_length=MAX_LENGTH,
        stride=0,
        strategy="longest_first",
    )

## Definition du Padding vecteur [PAD] pour que les sequences soient de meme longueur a usage avec attention_mask
def configure_padding(
    tokenizer: Tokenizer,
    max_length: int = MAX_LENGTH,
) -> None:
    pad_id = tokenizer.token_to_id("[PAD]")

    if pad_id is None:
        raise ValueError(
            "[PAD] absent du vocabulaire."
        )

    tokenizer.enable_padding(
        direction="right",
        pad_id=pad_id,
        pad_token="[PAD]",
        length=max_length,
    )