from torch.utils.data._utils.collate import default_collate
from dataloaders.scotus_attribute_mapping import ISSUE_AREA_MAPPING, KEYWORD_2_PETITIONER_NAME, PARTY_WINNING_MAPPING, PETITIONER_MAPPING, RESPONDENT_MAPPING, STATE_2_REGION_MAP
from configs.supported import F1Custom, binary_logits_to_pred_v2
from typing import Counter, List, Tuple, Union
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.datasets.amazon_dataset import AmazonDataset
import os
import jsonlines
import pandas as pd
import torch
import re
from wilds.common.metrics.all_metrics import F1, multiclass_logits_to_pred
from utils import get_info_logger, read_jsonl
from wilds.common.grouper import CombinatorialGrouper
import numpy as np
from tqdm import tqdm



class ScotusDataset(WILDSDataset):
    _versions_dict = {
        "0.4": {
            "download_url": "https://sid.erda.dk/share_redirect/bAlosmBf2K",
            "compressed_size": 136_72,
        },
    }

    def __init__(
        self,
        split_scheme='official',
        group_by_fields:Union[Tuple, str]='decisionDirection',
        root_dir="data/datasets",
        version=None,
        download=False,
        **kwargs,
    ):
        if isinstance(group_by_fields, str):
            group_by_fields = (group_by_fields,)
        if isinstance(group_by_fields, list):
            group_by_fields = tuple(group_by_fields)
        self._dataset_name = "scotus"
        self._version = version
        self.label2idx = ISSUE_AREA_MAPPING
        self._metadata_fields = group_by_fields#['decisionDirection', 'respondent']

        self._y_size = len(self.label2idx)
        self._n_classes = len(self.label2idx)
        
        self.group_by_fields = group_by_fields
        
        self.logger = get_info_logger(__name__)
        self._split_scheme = split_scheme
        self._data_dir = self.initialize_data_dir(root_dir, download=download)
        self.logger.info("reading data")
        self.data_df, self.attribute2value2idx = self.read_data()

        print(self.data_df.head())

        # Get the y values
        self._y_type = 'long'
        self._y_array = torch.LongTensor(self.data_df["labels"])
        # self._collate = self.__collate

        # Extract text
        self._text_array = list(self.data_df["text"])

        self._split_dict = {"train": 0, "val": 1, "test": 2}
        self._split_names = {"train": "train", "val": "dev", "test": "test"}

        for split in self._split_dict:
            split_indices = self.data_df["data_type"] == self._split_names[split]
            self.data_df.loc[split_indices, "data_type"] = self._split_dict[split]

        df = self.data_df
        self._split_array = self.data_df["data_type"].values

        # Extract metadata
        self._metadata_array = torch.LongTensor(self.data_df.loc[:, self._metadata_fields].values)

        self.initialize_eval_grouper()

        self._metric = F1(average="macro")

        super().__init__(root_dir, download, split_scheme)

    # def __collate(self, examples):
    #     inputs, targets, metadata = default_collate(examples)
    #     new_targets = torch.zeros(len(targets), self._y_size)
    #     new_targets[torch.arange(0, new_targets.size(0)).long(), targets.unsqueeze(0)] = 1
    #     new_targets = new_targets.clone()
    #     return inputs, new_targets, metadata
        

    def initialize_eval_grouper(self):
        self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=self.group_by_fields
            )

    def read_data(self):

        train_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.train.jsonl"),
            data_type="train",
            attributes_to_retain=self._metadata_fields,
        )
        dev_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.dev.jsonl"),
            data_type="dev",
            attributes_to_retain=self._metadata_fields,
        )
        test_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.test.jsonl"),
            data_type="test",
            attributes_to_retain=self._metadata_fields,
        )

        data = train_data + dev_data + test_data
        attribute2value2idx = dict()
        if 'respondent' in self._metadata_fields:
            data = self.map_respondent(data)
        for example in data:
            for a in self._metadata_fields:
                v = example[a]
                a_vals = attribute2value2idx.get(a, set())
                if str(v) == 'nan':
                    RuntimeError("Found nan as value for the protected attribute.")
                a_vals.add(v)
                attribute2value2idx[a] = a_vals

        for a in self._metadata_fields:
            a_vals = sorted(attribute2value2idx[a])
            v2i = dict(reversed(x) for x in enumerate(a_vals))
            del attribute2value2idx[a]
            attribute2value2idx[a] = v2i
        self.protected_attribute_id2str = dict()


        for example in data:
            for a in self.group_by_fields:
                str_val = example[a]
                idx_val = attribute2value2idx[a][example[a]]
                example[a] = idx_val
                example[a + "_str"] = str_val
                self.protected_attribute_id2str[idx_val] = str_val
            labels = [0] * self._n_classes
            labels[self.label2idx[example['label']]] = 1
            example['label'] = labels
            example['labels'] = example['label']

        if self.split_scheme == "official":
            df = pd.DataFrame(data)
        elif self.split_scheme == "temporal":
            df = self.__create_temporal_split(data, len(train_data), len(dev_data))
        elif self.split_scheme == "uniform":
            df = self.__create_uniform_split(data)
        df = df.fillna("")
        return df, attribute2value2idx

    def map_respondent(self, data):
        for example in data:
            pa_val = str(example['respondent']).lower()
            new_val = RESPONDENT_MAPPING[pa_val]
            example['respondent'] = new_val
        return data

    def get_input(self, idx):
        return self._text_array[idx]

    def eval(self, y_pred, y_true, metadata):
        metric = F1Custom(average='macro', prediction_fn=multiclass_logits_to_pred, target_fn=multiclass_logits_to_pred)
        eval_grouper = self._eval_grouper

        return self.standard_group_eval(metric, eval_grouper, y_pred, y_true, metadata)

    def get_stats(self):
        train_label_distr = Counter(self.data_df['label_id'][self.data_df['data_type']==0])
        dev_label_distr = Counter(self.data_df['label_id'][self.data_df['data_type']==1])
        test_label_distr = Counter(self.data_df['label_id'][self.data_df['data_type']==2])
        train_group_distr = Counter(self.data_df[self.protected_attribute][self.data_df['data_type']==0])
        dev_group_distr = Counter(self.data_df[self.protected_attribute][self.data_df['data_type']==1])
        test_group_distr = Counter(self.data_df[self.protected_attribute][self.data_df['data_type']==2])
        print(f'Train label distr: {train_label_distr.most_common(len(train_label_distr))}')
        print(f'Dev label distr: {dev_label_distr.most_common(len(dev_label_distr))}')
        print(f'Test label distr: {test_label_distr.most_common(len(test_label_distr))}')
        
        print(f'Train group distr: {train_group_distr.most_common(len(train_group_distr))}')
        print(f'Dev group distr: {dev_group_distr.most_common(len(dev_group_distr))}')
        print(f'Test group distr: {test_group_distr.most_common(len(test_group_distr))}')

    @staticmethod        
    def dump_dataset(scotus_dataset:WILDSDataset, outfile):
        training_set = scotus_dataset.get_subset('train')
        dev_set = scotus_dataset.get_subset('val')
        bar = tqdm(total=len(training_set)+len(dev_set))
        with open(outfile, 'w') as out_file:
            for dataset in [training_set, dev_set]:
                for example in dataset:
                    bar.update(1)
                    text = example[0]
                    out_file.write(' </s> '.join(re.split('\n{2,}', text)).replace('\n', '') + '\n')

    @staticmethod
    def __load_predictions(path):
        predictions = list()
        with open(path) as lines:
            for line in lines:
                pred = np.array([float(x) for x in line.strip().split(',')])
                predictions.append(pred)
        predictions = np.stack(predictions, 0)
        return np.argmax(predictions, -1)
    @staticmethod
    def load_raw_dataset(path):
        seen = set()
        data = []
        with jsonlines.open(path) as lines:
            for line_data in lines:
                example = dict(line_data)
                if example['attributes']['caseId'] in seen:
                    continue
                seen.add(example['attributes']['caseId'])
                data.append(example)
        return data

    @staticmethod
    def dump_distilled_dataset(data_dire, logdir:str, seed:int, outdir:str):
        train_data, dev_data, test_data = [], []
        seen = set()
        train_data = load_raw_dataset(os.path.join(data_dir, "scotus.train.jsonl"))
        dev_data = load_raw_dataset(os.path.join(data_dir, "scotus.dev.jsonl"))
        test_data = load_raw_dataset(os.path.join(data_dir, "scotus.test.jsonl"))
        

        val_pred = ScotusDataset.__load_predictions(os.path.join(logdir, f'scotus_split:val_seed:{seed}_epoch:best_pred.csv'))
        test_pred = ScotusDataset.__load_predictions(os.path.join(logdir, f'scotus_split:test_seed:{seed}_epoch:best_pred.csv'))
    @staticmethod
    def dump_error_dataset(data_dir, logdir:str, seed:int, outdir:str, attribute_filter:tuple=None):
        dev_data, test_data = [], []
        seen = set()
        with jsonlines.open(os.path.join(data_dir, "scotus.dev.jsonl")) as lines:
            for line_data in lines:
                example = dict(line_data)
                if example['attributes']['caseId'] in seen:
                    continue
                seen.add(example['attributes']['caseId'])
                dev_data.append(example)
        seen = set()
        with jsonlines.open(os.path.join(data_dir, "scotus.test.jsonl")) as lines:
            for line_data in lines:
                example = dict(line_data)
                if example['attributes']['caseId'] in seen:
                    continue
                seen.add(example['attributes']['caseId'])
                test_data.append(example)
        
        val_pred = ScotusDataset.__load_predictions(os.path.join(logdir, f'scotus_split:val_seed:{seed}_epoch:best_pred.csv'))
        test_pred = ScotusDataset.__load_predictions(os.path.join(logdir, f'scotus_split:test_seed:{seed}_epoch:best_pred.csv'))
        
        val_true = np.array([x['label_id'] for x in dev_data])
        test_true = np.array([x['label_id'] for x in test_data])

        val_errors = val_pred != val_true
        test_errors = test_pred != test_true
        
        dev_data = np.array(dev_data)[val_errors].tolist()
        test_data = np.array(test_data)[test_errors].tolist()
        if attribute_filter is not None:
            att_name, att_value = attribute_filter
            dev_data = list(filter(lambda example: example['attributes'][att_name] == att_value, dev_data))
            test_data = list(filter(lambda example: example['attributes'][att_name] == att_value, test_data))
        dataset = dev_data + test_data
        with jsonlines.open(os.path.join(outdir, 'scotus.train.jsonl'), 'w') as writer_train, jsonlines.open(os.path.join(outdir, 'scotus.dev.jsonl'), 'w') as writer_dev, jsonlines.open(os.path.join(outdir, 'scotus.test.jsonl'), 'w') as writer_test:
            for o in dataset:
                writer_train.write(o)
                writer_dev.write(o)
                writer_test.write(o)
        open(os.path.join(outdir,'RELEASE_v0.4.txt'), 'w').close()

if __name__ == "__main__":
    # dataset = ScotusDataset()

    ScotusDataset.dump_error_dataset('data/datasets/scotus_v0.4', 
    '/home/npf290/dev/fairlex-wilds/logs_final_tfidf_regressor/scotus/ERM/respondent/seed_1', 
    1, 
    '/home/npf290/dev/fairlex-wilds/data/linear_interpreter_datasets_decision-direction=liberal_seed_1/scotus_v0.4',
    ('decisionDirection', 'liberal'))
    # dump_dataset(dataset, "data/scotus_v0.4/dump_temporal_train_dev.txt")
    # dataset.get_stats()
    exit(1)

    print('Uniform - decisionDirection')
    dataset =ScotusDataset("uniform", "decisionDirection")
    dataset.get_stats()
    print('Temporal - decisionDirection')
    dataset =ScotusDataset("temporal", "decisionDirection")
    dataset.get_stats()
    print('Offficial - decisionDirection')
    dataset = ScotusDataset("official", "decisionDirection")
    dataset.get_stats()
    print('='*100)

    print('Uniform - issueArea')
    dataset =ScotusDataset("uniform", "issueArea")
    dataset.get_stats()
    print('Temporal - issueArea')
    dataset =ScotusDataset("temporal", "issueArea")
    dataset.get_stats()
    print('Official - issueArea')
    dataset =ScotusDataset("official", "issueArea")
    dataset.get_stats()
    
