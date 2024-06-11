from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.text.datasets import multi30k, Multi30k
from typing import Iterable, List

multi30k.URL['train'] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL['valid'] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TRG_LANGUAGE = 'en'

token_transform = {}
vocab_transform = {}


