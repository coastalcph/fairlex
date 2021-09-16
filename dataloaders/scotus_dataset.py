from dataloaders.scotus_attribute_mapping import ISSUE_AREA_MAPPING, KEYWORD_2_PETITIONER_NAME, PARTY_WINNING_MAPPING, PETITIONER_MAPPING, RESPONDENT_MAPPING, STATE_2_REGION_MAP
from configs.supported import binary_logits_to_pred_v2
from typing import Counter
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.datasets.amazon_dataset import AmazonDataset
import os
import jsonlines
import pandas as pd
import torch
import re
from wilds.common.metrics.all_metrics import F1
from utils import get_info_logger, read_jsonl
from wilds.common.grouper import CombinatorialGrouper
from torch.nn.utils.rnn import pad_sequence
from random import shuffle
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
        split_scheme,
        protected_attribute,
        protected_attribute_val=None,
        root_dir="data/",
        version=None,
        download=False,
        **kwargs,
    ):
        self._dataset_name = "scotus"
        self._version = version
        # self.prediction_fn = binary_logits_to_pred_v2
        words = protected_attribute.split("_")
        for i in range(1, len(words)):
            words[i] = words[i][0].upper() + words[i][1:]
        self.label2idx = ISSUE_AREA_MAPPING
        protected_attribute = "".join(words)
        self.protected_attribute = protected_attribute
        self.protected_attribute_val = protected_attribute_val.replace('_', ' ') if protected_attribute_val is not None else None
        self.make_protected_group_binary = protected_attribute_val is not None

        self.logger = get_info_logger(__name__)
        self.min_threshold = kwargs.get("min_example_threshold", 500)
        self.attributes_to_retain = list(set(["dateDecision", self.protected_attribute]))
        self._split_scheme = split_scheme
        self._data_dir = self.initialize_data_dir(root_dir, download=download)
        self.logger.info("reading data")
        self._metadata_df, self.attribute2value2idx = self.read_data()

        # print(self._metadata_df.head())

        # Get the y values
        self._y_type = 'long'
        self._y_array = torch.LongTensor(self._metadata_df["label_id"])
        self._y_size = len(self.label2idx)
        self._n_classes = None

        # Extract text
        self._text_array = list(self._metadata_df["text"])

        self._split_dict = {"train": 0, "val": 1, "test": 2}
        self._split_names = {"train": "train", "val": "dev", "test": "test"}

        for split in self._split_dict:
            split_indices = self._metadata_df["data_type"] == self._split_names[split]
            self._metadata_df.loc[split_indices, "data_type"] = self._split_dict[split]
        self._split_array = self._metadata_df["data_type"].values

        # Extract metadata

        self._metadata_array = torch.cat(
            (
                torch.LongTensor(
                    self._metadata_df.loc[:, self.attributes_to_retain].values
                ),
                self._y_array.reshape((-1, 1)),
            ),
            dim=1,
        )
        self._metadata_fields = self.attributes_to_retain + ["y"]

        self.initialize_eval_grouper()

        self._metric = F1(average="macro")

        super().__init__(root_dir, download, split_scheme)

    def initialize_eval_grouper(self):
        self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=[self.protected_attribute]
            )

    def read_data(self):

        train_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.train.jsonl"),
            data_type="train",
            attributes_to_retain=self.attributes_to_retain,
        )
        dev_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.dev.jsonl"),
            data_type="dev",
            attributes_to_retain=self.attributes_to_retain,
        )
        test_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.test.jsonl"),
            data_type="test",
            attributes_to_retain=self.attributes_to_retain,
        )

        data = train_data + dev_data + test_data
        attribute2value2idx = dict()
        if self.protected_attribute == 'respondent':
            data = self.map_respondent(data)
        for example in data:
            for a in self.attributes_to_retain:
                v = example[a]
                if self.make_protected_group_binary and a == self.protected_attribute:
                    
                    if v != self.protected_attribute_val:
                        v = "Not " + self.protected_attribute_val
                        example[a] = v
                a_vals = attribute2value2idx.get(a, set())
                a_vals.add(v)
                attribute2value2idx[a] = a_vals

        for a in self.attributes_to_retain:
            a_vals = sorted(attribute2value2idx[a])
            v2i = dict(reversed(x) for x in enumerate(a_vals))
            del attribute2value2idx[a]
            attribute2value2idx[a] = v2i
        self.protected_attribute_id2str = dict()


        for example in data:
            for a in self.attributes_to_retain:
                str_val = example[a]
                idx_val = attribute2value2idx[a][example[a]]
                example[a] = idx_val
                example[a + "_str"] = str_val
                if a == self.protected_attribute:
                    self.protected_attribute_id2str[idx_val] = str_val

        if self.split_scheme == "official":
            df = pd.DataFrame(data)
            df.fillna("")
        elif self.split_scheme == "temporal":
            df = self.__create_temporal_split(data, len(train_data), len(dev_data))
        elif self.split_scheme == "uniform":
            df = self.__create_uniform_split(data)
        # df = df.fillna("")
        return df, attribute2value2idx

    def map_respondent(self, data):
        for example in data:
            pa_val = example[self.protected_attribute].lower()
            new_val = RESPONDENT_MAPPING[pa_val]
            example[self.protected_attribute] = new_val
        return data
        
    def map_petitioners(self, data):
        for example in data:
            pa_val = example[self.protected_attribute].lower()
            for keyword_regex in KEYWORD_2_PETITIONER_NAME.keys():
                if re.match(keyword_regex, pa_val) is not None:
                    example[self.protected_attribute] = KEYWORD_2_PETITIONER_NAME[keyword_regex]
            
    def coarsen_case_origin(self, data):
        found_count = 0
        notfound_count = 0
        for example in data:
            pa_val = example[self.protected_attribute].lower()
            found = False
            for state in STATE_2_REGION_MAP.keys():
                if state in pa_val:
                    region = STATE_2_REGION_MAP[state]
                    example[self.protected_attribute] = region
                    found = True
                    found_count += 1

            if not found:
                notfound_count += 1
        print()
                

    def __create_temporal_split(self, data, train_size, dev_size):
        data = sorted(data, key=lambda ex: ex["dateDecision"])
        for i, ex in enumerate(data):
            if i < train_size:
                ex["data_type"] = "train"
            elif i < train_size + dev_size:
                ex["data_type"] = "dev"
            else:
                ex["data_type"] = "test"

        return pd.DataFrame(data)

    def __create_uniform_split(self, data):
        data = [ex for ex in data if ex["label_id"] in {0, 1}]
        groups = self.__group_list_by(data, self.protected_attribute)
        included_groups = {
            k: v for k, v in groups.items() if len(v) >= self.min_threshold
        }
        excluded_groups = {
            self.protected_attribute_id2str[k]: len(v)
            for k, v in groups.items()
            if len(v) < self.min_threshold
        }
        groups = included_groups
        # groups = self.__balance_groups(groups)
        min_size = min([len(g) for g in groups.values()])
        train_data = []
        dev_data = []
        test_data = []
        group_size = min_size // 3
        train_perc, dev_perc, test_perc = 0.8, 0.1, 0.1
        for k, v in groups.items():
            shuffle(v)
            v = v[:group_size]

            train_size = int(len(v) * train_perc)
            dev_size = int(len(v) * dev_perc)
            test_size = int(len(v) * test_perc)
            train_data.extend(v[:train_size])
            dev_data.extend(v[train_size : train_size + dev_size])
            test_data.extend(
                v[train_size + dev_size : train_size + dev_size + test_size]
            )

        train_groups = self.__group_list_by(train_data, self.protected_attribute)
        dev_groups = self.__group_list_by(dev_data, self.protected_attribute)
        test_groups = self.__group_list_by(test_data, self.protected_attribute)
        assert len({len(v) for k, v in train_groups.items()}) == 1
        assert len({len(v) for k, v in dev_groups.items()}) == 1
        assert len({len(v) for k, v in test_groups.items()}) == 1

        for ex in train_data:
            ex["data_type"] = "train"
        for ex in dev_data:
            ex["data_type"] = "dev"
        for ex in test_data:
            ex["data_type"] = "test"
        return pd.DataFrame(train_data + dev_data + test_data)

    def __group_list_by(self, lst, key):
        grouped_by = dict()
        for x in lst:
            key_val = x[key]
            group_lst = grouped_by.get(key_val, list())
            group_lst.append(x)
            grouped_by[key_val] = group_lst
        return grouped_by

    def get_input(self, idx):
        return self._text_array[idx]

    def eval(self, y_pred, y_true, metadata):
        metric = F1(average='micro')
        eval_grouper = self._eval_grouper

        return self.standard_group_eval(metric, eval_grouper, y_pred, y_true, metadata)

    def get_stats(self):
        train_label_distr = Counter(self._metadata_df['label_id'][self._metadata_df['data_type']==0])
        dev_label_distr = Counter(self._metadata_df['label_id'][self._metadata_df['data_type']==1])
        test_label_distr = Counter(self._metadata_df['label_id'][self._metadata_df['data_type']==2])
        train_group_distr = Counter(self._metadata_df[self.protected_attribute][self._metadata_df['data_type']==0])
        dev_group_distr = Counter(self._metadata_df[self.protected_attribute][self._metadata_df['data_type']==1])
        test_group_distr = Counter(self._metadata_df[self.protected_attribute][self._metadata_df['data_type']==2])
        print(f'Train label distr: {train_label_distr.most_common(len(train_label_distr))}')
        print(f'Dev label distr: {dev_label_distr.most_common(len(dev_label_distr))}')
        print(f'Test label distr: {test_label_distr.most_common(len(test_label_distr))}')
        
        print(f'Train group distr: {train_group_distr.most_common(len(train_group_distr))}')
        print(f'Dev group distr: {dev_group_distr.most_common(len(dev_group_distr))}')
        print(f'Test group distr: {test_group_distr.most_common(len(test_group_distr))}')
        
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
    

if __name__ == "__main__":
    dataset = ScotusDataset("temporal", "decisionDirection", download=True)
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
    
