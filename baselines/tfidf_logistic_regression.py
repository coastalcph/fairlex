from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import jsonlines
import pandas as pd
import re
import numpy as np
from scipy import sparse as scisparse
import os
import logging
import sys
import pickle as pkl

class TfIdfLogisticRegression:
    def __init__(
        self, train_path, dev_path, test_path, max_features, protected_attribute
    ) -> None:
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.protected_attribute = protected_attribute
        self.training_set: pd.DataFrame = self.load_dataset(train_path)
        self.dev_set: pd.DataFrame = self.load_dataset(dev_path)
        self.test_set: pd.DataFrame = self.load_dataset(test_path)
        self.text_transformer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            max_features=max_features,
        )
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self._maybe_learn_tfidf()
        
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
            self.logger.info("No precomputed tfidf dataset found. Learning from data...")
            if is_training:
                dataset = self.text_transformer.fit_transform(data["text"])
            else: dataset = self.text_transformer.transform(data['text'])
            self.logger.info(f"Saving tfidf dataset @ {path}")
            scisparse.save_npz(path, dataset)
            if is_training:
                with open(tokenizer_path, 'wb') as writer:
                    pkl.dump(self.text_transformer, writer)

        return dataset

    def _maybe_learn_tfidf(self):
        tfidf_training_path = self.train_path + '.tfidf.npz'
        tfidf_dev_path = self.dev_path + '.tfid.npz'
        tfidf_test_path = self.test_path + '.tfidf.npz'
        self.tfidf_training = self._maybe_load_dataset(tfidf_training_path, self.training_set, True)
        self.tfidf_dev = self._maybe_load_dataset(tfidf_dev_path, self.dev_set, False)
        self.tfidf_test = self._maybe_load_dataset(tfidf_test_path, self.test_set, False)

    def load_dataset(self, path):
        data = list()
        with jsonlines.open(path) as lines:
            for line in lines:
                text = re.sub('\n *\n+', '\n', line["text"])
                label = line["label"]
                protected_attribute_val = line["attributes"][self.protected_attribute]

                data.append(
                    {
                        "text": text,
                        "label": label,
                        self.protected_attribute: protected_attribute_val,
                    }
                )
        return pd.DataFrame(data)

    def learn(self):
        self.logger.info("Training...")
        self.regressor.fit(self.tfidf_training, self.training_set["label"])
        from sklearn import metrics
        predictions = self.regressor.predict(self.tfidf_dev)
        metrics.f1_score(predictions, self.dev_set['label'], average='micro')
        print()


if __name__ == "__main__":
    model = TfIdfLogisticRegression(
        "data/scotus_v0.4/scotus.train.jsonl",
        "data/scotus_v0.4/scotus.dev.jsonl",
        "data/scotus_v0.4/scotus.test.jsonl",
        max_features=10_000,
        protected_attribute="decisionDirection",
    )
    model.learn()
