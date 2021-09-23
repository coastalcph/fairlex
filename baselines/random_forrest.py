from dataloaders import get_dataset
from random import Random
from dataloaders.scotus_dataset import ScotusDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from wilds.datasets.wilds_dataset import WILDSDataset
import pandas as pd
from baselines.eval import eval, eval_by_group
from pprint import pprint
import pickle as pkl
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RandomForest():
    def __init__(self, max_depth = 100, random_state=0, num_classes=1) -> None:
        self.random_state = random_state
        self.max_depth = max_depth
        if num_classes == 1:
            self.classifier = RandomForestClassifier(max_depth=100, random_state=random_state)
        else: 
            self.classifier = RandomForestClassifier(max_depth=max_depth, random_state=random_state, class_weight=[{0:1,1:1} for _ in range(num_classes)])

    def train(self, dataset:pd.DataFrame, text_featurizer:TfidfVectorizer):
        Y = dataset['labels']
        X = dataset['text']
        X = text_featurizer.transform(X)
        self.classifier.fit(X, np.stack(Y))

    def forward(self, dataset:pd.DataFrame, text_featurizer:TfidfVectorizer):
        X = text_featurizer.transform(dataset['text'])
        return self.classifier.predict(X)

    @classmethod
    def from_pretrained(cls, path):
        if not os.path.exists(path):
            logger.warning(f'{path} does not exists')
            raise RuntimeError(f'{path} does not exists')
        rf = RandomForest()
        with open(path, 'rb') as reader:
            classifier = pkl.load(reader)
        rf.classifier = classifier
        return rf

    def save(self, path):
        with open(path, 'wb') as writer:
            pkl.dump(self.classifier, writer)




if __name__ == '__main__':
    num_features = 10_000
    random_state = 0
    max_depth = 100
    dataset_name = 'ecthr'
    # num_classes = 11
    num_classes = 14 #etchr
    # vectorizer_path = '/home/npf290/dev/fairlex-wilds/data/datasets/scotus_v0.4/tfidf_tokenizer.pkl'
    vectorizer_path = 'data/datasets/ecthr_v1.0/tfidf_tokenizer.pkl'
    model_path = f'data/models/random_forest_max-depth={max_depth}_random-state={random_state}.{dataset_name}.pkl'
    attribute_name = 'defendant'
    train_model = False
    try:
        rf_classifier = RandomForest.from_pretrained(model_path)
    except:
        rf_classifier = RandomForest(num_classes=num_classes)
        train_model = True
    train_featurizer = False
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as reader:
            text_featurizer = pkl.load(reader)

    else:
        train_featurizer = True
        text_featurizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                lowercase=True,
                max_features=num_features,
            )
    
    original_dataset = get_dataset(dataset_name, group_by_fields=(attribute_name, ))
    dataset = original_dataset.data_df
    train = dataset[dataset['data_type'] == 0]
    dev = dataset[dataset['data_type'] == 1]
    test = dataset[dataset['data_type'] == 2]
    if train_featurizer:
        print('Training featurizer')
        text_featurizer.fit(train['text'])
        with open(vectorizer_path, 'wb') as writer:
            pkl.dump(text_featurizer, writer)
    if train_model:        
        print('Training model')
        rf_classifier.train(train, text_featurizer)
        rf_classifier.save(model_path)
    
    print('=' * 150)
    print('Eval')
    print('=' * 150)
    print(eval(rf_classifier.classifier, test, text_featurizer, test['labels']))

    print('=' * 150)
    print('Eval By Group')
    print('=' * 150)
    
    # mapping = {k:v for k, v in test[['respondent', 'respondent_str']].values.tolist()}
    pprint(eval_by_group(rf_classifier.classifier, test, attribute_name, text_featurizer))
    
    