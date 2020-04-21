import tokenizers
import pathlib


DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'
BERT_PATH = DATA_PATH / 'bert_base_uncased'
ELECTRA_PATH = DATA_PATH / 'electra-base'
TRAIN_FOLDS = DATA_PATH / 'train_folds.csv'
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = pathlib.Path(__file__).parent.parent / 'models'
TRAIN_FILE = DATA_PATH / 'train_fold4.csv'
VALID_FILE = DATA_PATH / 'valid_fold4.csv'

TOKENIZER_BERT = tokenizers.BertWordPieceTokenizer(
    f"{str(BERT_PATH / 'vocab.txt')}",
    lowercase=True
)

TOKENIZER_ELECTRA = tokenizers.BertWordPieceTokenizer(
    f"{str(ELECTRA_PATH / 'vocab.txt')}",
    lowercase=True
)
