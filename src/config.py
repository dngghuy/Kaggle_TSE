import tokenizers
import pathlib


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'
BERT_PATH = DATA_PATH / 'bert_base_uncased'
MODEL_PATH = "model.bin"
TRAIN_FILE = DATA_PATH / 'train.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"{str(BERT_PATH / 'vocab.txt')}",
    lowercase=True
)

