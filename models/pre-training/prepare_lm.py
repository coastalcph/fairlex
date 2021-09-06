from transformers import AutoTokenizer, AutoConfig
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import BertProcessing
from data import DATA_DIR, MODELS_DIR
import os

ECTHR_FILE = os.path.join(DATA_DIR, 'datasets', 'ecthr_v1.0', 'ecthr.text.raw')
HIDDEN_SIZE = 256
INT_SIZE = 1024
NUM_ATTENTION_HEADS = 4
N_LAYERS = 4
ATTENTION_WINDOW = 256
MAX_SEQ_LENGTH = 4098
VOCAB_SIZE = 25000

# TRAIN TOKENIZER
tokenizer = Tokenizer(WordPiece())
tokenizer.pre_tokenizer = BertPreTokenizer()
tokenizer.normalizer = BertNormalizer()
tokenizer.post_processor = BertProcessing(sep=('</s>', 1), cls=('<s>', 0))
trainer = WordPieceTrainer(special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>'], vocab_size=25000)
tokenizer.train(files=[ECTHR_FILE], trainer=trainer)
tokenizer.model_max_length = MAX_SEQ_LENGTH
tokenizer.save(os.path.join(MODELS_DIR, 'lex-longformer', 'tokenizer.json'))

# BUILD CONFIGURATION FILE
config = AutoConfig.from_pretrained('allenai/longformer-base-4096')

config.hidden_size = HIDDEN_SIZE
config.intermediate_size = INT_SIZE
config.num_attention_heads = NUM_ATTENTION_HEADS
config.max_position_embeddings = MAX_SEQ_LENGTH
config.vocab_size = VOCAB_SIZE
config.attention_window = [ATTENTION_WINDOW] * N_LAYERS
config.num_hidden_layers = N_LAYERS
config.bos_token_id = tokenizer.token_to_id('<s>')
config.sep_token_id = tokenizer.token_to_id('</s>')
config.eos_token_id = tokenizer.token_to_id('</s>')
config.mask_token_id = tokenizer.token_to_id('<mask>')
config.unk_token_id = tokenizer.token_to_id('<unk>')
config.model_max_length = MAX_SEQ_LENGTH
config.save_pretrained(os.path.join(MODELS_DIR, 'lex-longformer'))

tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, 'lex-longformer'))
print()