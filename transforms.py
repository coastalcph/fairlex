import re

import nltk
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import os
from data import DATA_DIR
from nltk.corpus import stopwords
import thulac
import stopwordsiso

def initialize_transform(transform_name, config):
    if transform_name is None:
        return None
    elif transform_name == 'bert':
        return initialize_bert_transform(config)
    elif transform_name == 'hier-bert':
        return initialize_hierbert_transform(config)
    elif transform_name == 'tf-idf':
        return initialize_tfidf(config)
    else:
        raise ValueError(f"{transform_name} not recognized")


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


def initialize_hierbert_transform(config):
    assert 'bert' in config.model
    assert config.max_segment_length is not None
    assert config.max_segments is not None

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    def transform(text):
        paragraphs = []
        paragraphs_tokens = {'input_ids': torch.zeros(config.max_segments, config.max_segment_length,
                                                      dtype=torch.int32),
                             'attention_mask': torch.zeros(config.max_segments, config.max_segment_length,
                                                           dtype=torch.int32)}

        for idx, paragraph in enumerate(text.split('</s>')[:config.max_segments]):
            paragraphs.append(paragraph)

        tokens = tokenizer(
            paragraphs,
            padding='max_length',
            truncation=True,
            max_length=config.max_segment_length,
            return_tensors='pt')

        paragraphs_tokens['input_ids'][:len(paragraphs)] = tokens['input_ids']
        paragraphs_tokens['attention_mask'][:len(paragraphs)] = tokens['attention_mask']

        x = torch.stack(
            (paragraphs_tokens['input_ids'],
             paragraphs_tokens['attention_mask']),
            dim=2)
        # x = torch.squeeze(x, dim=0)
        return x
    return transform


def initialize_tfidf(config):
    cut = thulac.thulac(seg_only=True)
    def preprocess_text(text: str):
        return re.sub('[0-9]+', ' ', text)

    def tokenize(text: str):
        if config.dataset in ['ecthr', 'scotus']:
            return nltk.word_tokenize(text)
        elif config.dataset == 'spc':
            return cut.cut(text, text=True)
        else:
            return nltk.word_tokenize(text, language='german')

    if config.dataset in ['ecthr', 'scotus']:
        stop_words = set(stopwords.words('english'))
    elif config.dataset == 'spc':
        stop_words = None #set(list(stopwordsiso.stopwords("zh")) + ['借傥', '兼', '前', '唷', '啪', '啷', '喔', '天', '始', '漫', '然', '特', '竟', '莫', '见', '设', '达'])
    else:
        stop_words = set(stopwords.words('german') + stopwords.words('french') + stopwords.words('italian'))

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, stop_words=stop_words,
                                 preprocessor=preprocess_text, lowercase=False, tokenizer=tokenize, min_df=5)
    with open(os.path.join(DATA_DIR, 'datasets', f'{config.dataset}_v1.0', f'{config.dataset}_dump.txt')) as file:
        vectorizer.fit(file.readlines())
    print('VECTORIZER TRAINED !!!')

    def transform(text):
        text = ' '.join(text.replace('</s>', ' ').split()[:config.max_token_length])
        x = torch.as_tensor(vectorizer.transform([text]).todense()).float()
        x = torch.squeeze(x, dim=0)
        return x
    return transform
