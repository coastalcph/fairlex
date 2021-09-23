from dataloaders.ecthr_dataset import ECtHRDataset
from dataloaders.scotus_dataset import ScotusDataset
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import jsonlines
import pandas as pd
import re
import scipy
import numpy as np
from scipy import sparse as scisparse
import os
import logging
import sys
import pickle as pkl
from scipy.sparse import csr_matrix

class TfIdfLogisticRegression:
    def __init__(
        self,max_features
    ) -> None:
        
        self.__cache_path = ".cache/"
        if not os.path.exists(self.__cache_path):
            os.makedirs(self.__cache_path)
        self.text_transformer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            lowercase=True,
            max_features=max_features,
        )
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.regressor = LogisticRegression(
            C=5e1, solver="lbfgs", multi_class="multinomial", random_state=17, n_jobs=16,
            max_iter=100, verbose=1
        )
    
    def _maybe_load_dataset(self, path, data, is_training):
        root = '/'.join(path.split('/')[:-1])
        tokenizer_path = os.path.join(root, 'tfidf_tokenizer.pkl')
                
        if os.path.exists(path):
            self.logger.info(f"Found dataset at {path}. Loading...")
            dataset = scisparse.load_npz(path)
            if is_training:
                with open(tokenizer_path, 'rb') as reader:
                    self.text_transformer = pkl.load(reader)
        else:
            if is_training:
                self.logger.info("No precomputed tfidf dataset found. Learning from data...")
                dataset = self.text_transformer.fit_transform(data["text"])
            else: dataset = self.text_transformer.transform(data['text'])
            self.logger.info(f"Saving tfidf dataset @ {path}")
            scisparse.save_npz(path, dataset)
            if is_training:
                with open(tokenizer_path, 'wb') as writer:
                    pkl.dump(self.text_transformer, writer)

        return dataset

    def _maybe_learn_tfidf(self, dataset, dataset_name):
        tfidf_training_path = self.__cache_path + f'{dataset_name}.tfidf.npz'
        # tfidf_dev_path = self.__cache_path + f'{dataset_name}.dev.tfidf.npz'
        # tfidf_test_path = self.__cache_path + f'{self.dataset_name}.test.tfidf.npz'
        return self._maybe_load_dataset(tfidf_training_path, dataset, True)
        # self.tfidf_dev = self._maybe_load_dataset(tfidf_dev_path, self.dev_set, False)
        # self.tfidf_test = self._maybe_load_dataset(tfidf_test_path, self.test_set, False)


    def train(self, training_set, dataset_name):
        self.logger.info("Training...")
        tfidf_training = self._maybe_learn_tfidf(training_set, dataset_name + ".train")
        labels = training_set['labels']
        
        labels_stack = np.stack(training_set['labels'].values)
        if len(labels_stack) > 1 and labels_stack.shape[1] > 1:
            labels  = scipy.sparse.csr_matrix(np.stack(training_set['labels'].values))
        self.regressor.fit(tfidf_training, labels)
    
    def eval_by_group(self, dataset:pd.DataFrame, protected_attribute_name:str, val_mapping:dict=None):
        f1_by_group = dict()
        for val in dataset[protected_attribute_name].unique():
            group_dataset = dataset[dataset[protected_attribute_name]==val]
            if val_mapping is not None:
                val = val_mapping.get(val, val)
            f1_by_group[val] = self.eval(group_dataset)
        return f1_by_group

    def eval(self, dataset:Union[pd.DataFrame, csr_matrix], labels = None):
        from sklearn import metrics
        assert type(dataset) != csr_matrix or labels is not None
        if type(dataset) == pd.DataFrame:
            labels = dataset['label']
            dataset = self.text_transformer.transform(dataset['text'])
        predictions = self.regressor.predict(dataset)
        return metrics.f1_score(predictions,labels, average='micro')

if __name__ == "__main__":

    dataset_name = 'scotus'
    protected_attribute='respondent'
    if dataset_name == 'scotus':
            dataset = ScotusDataset("official", version="0.4", group_by_fields=(protected_attribute,))
    elif dataset_name == 'ecthr':
        dataset = ECtHRDataset(version="1.0")
    else:
        raise RuntimeError(f'The dataset `{dataset_name}` is not recognised.')

    model = TfIdfLogisticRegression(
        max_features=10_000,
    )
    training_set = dataset.data_df[dataset.data_df['data_type'] == dataset.split_dict['train']]
    model.train(training_set, dataset_name)

    test_set = dataset.data_df[dataset.data_df['data_type'] == dataset.split_dict['test']]

    print(model.eval_by_group(test_set, protected_attribute))
