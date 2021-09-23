import scipy
from transformers import AutoTokenizer
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle as pkl

def initialize_transform(transform_name, config):
    if transform_name is None:
        return None
    elif transform_name == 'bert':
        return initialize_bert_transform(config)
    elif transform_name == 'tfidf':
        return initialize_tfidf_transform(config)
    elif transform_name == 'bow':
        return initialize_bow_transform(config)
    else:
        raise ValueError(f"{transform_name} not recognized")

def initialize_bow_transform(config):
    assert hasattr(config, 'bow_vectorizer_path')
    with open(config.bow_vectorizer_path, 'rb') as reader:
        vectorizer: CountVectorizer = pkl.load(reader)
    def transform(text):
        sparse_matrix:csr_matrix = vectorizer.transform([text])
        npmatrix = sparse_matrix.todense()
        return torch.from_numpy(npmatrix).float().squeeze()
    return transform


def initialize_tfidf_transform(config):
    assert 'tfidf_vectorizer_path' in config.model_kwargs
    path = config.model_kwargs['tfidf_vectorizer_path']
    with open(path, 'rb') as reader:
        tfidf_vectorizer: TfidfVectorizer = pkl.load(reader)
    def transform(text):
        sparse_matrix:csr_matrix = tfidf_vectorizer.transform([text])
        npmatrix = sparse_matrix.todense()
        return torch.from_numpy(npmatrix).float().squeeze()
    return transform


def initialize_bert_transform(config):
    assert 'longformer' in config.model
    assert config.max_token_length is not None

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.max_token_length,
            return_tensors='pt')
        global_attention_mask = torch.zeros_like(tokens['input_ids'])
        # global attention on cls token
        global_attention_mask[:, 0] = 1
        # global attention to sep tokens
        global_attention_mask += (tokens['input_ids'] == tokenizer.sep_token_id).int()
        x = torch.stack(
            (tokens['input_ids'],
             tokens['attention_mask'],
             global_attention_mask),
            dim=2)
        x = torch.squeeze(x, dim=0)
        return x
    return transform

