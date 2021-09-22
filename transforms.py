from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import os
from data import DATA_DIR
from nltk.corpus import stopwords


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
    if config.dataset in ['ecthr', 'scotus']:
        stop_words = set(stopwords.words('english'))
        token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    else:
        stop_words = set(stopwords.words('german') + stopwords.words('french') + stopwords.words('italian'))
        token_pattern = r"(?u)\b[a-zA-ZÊõüèëØÅçéá̊òąàôœå≅êûßÖíä§ÂÇñǗÈËãøαÉÀĄïṣùâóÎÄúÓæöμÔìî]" \
                        r"[a-zA-ZÊõüèëØÅçéá̊òąàôœå≅êûßÖíä§ÂÇñǗÈËãøαÉÀĄïṣùâóÎÄúÓæöμÔìî]+\b"

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, stop_words=stop_words,
                                 token_pattern=token_pattern, lowercase=False)
    with open(os.path.join(DATA_DIR, 'datasets', f'{config.dataset}_v1.0', f'{config.dataset}_dump.txt')) as file:
        vectorizer.fit(file.readlines())
    print('VECTORIZER TRAINED !!!')

    def transform(text):
        text = ' '.join(text.replace('</s>', ' ').split()[:config.max_token_length])
        x = torch.as_tensor(vectorizer.transform([text]).todense()).float()
        x = torch.squeeze(x, dim=0)
        return x
    return transform
