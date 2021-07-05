from typing import Counter
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.datasets.amazon_dataset import AmazonDataset
import os
import jsonlines
import pandas as pd
import torch
from wilds.common.metrics.all_metrics import F1
from utils import get_info_logger, read_jsonl
from wilds.common.grouper import CombinatorialGrouper
from torch.nn.utils.rnn import pad_sequence


class ScotusDataset(WILDSDataset):
    _versions_dict = {
        '0.1': {
            'download_url': 'https://sid.erda.dk/share_redirect/G85ijt7lut',
            'compressed_size': 139_488
        },
    }
    def __init__(self, root_dir='data/', version=None, download=False, split_scheme="official", **kwargs):
        if 'groupby_fields' in kwargs:
            groupby_fields = kwargs['groupby_fields']
            if type(groupby_fields) == str:
                groupby_fields = groupby_fields.replace(', ', ',').split(',')
            elif type(groupby_fields) != list:
                raise RuntimeError(f'groupby_fields type ({type(groupby_fields)} not recognised.')
        else:
            groupby_fields = ['decisionDirection']
        self._dataset_name = "scotus"
        self._version = version
        self._identity_vars = [
            "issueArea",
            "decisionDirection"
        ]
        self._groupby_fields = groupby_fields
        self.logger = get_info_logger(__name__)

        self._split_scheme = split_scheme
        self._data_dir = self.initialize_data_dir(root_dir, download=download)
        self.logger.info("reading data")
        self._metadata_df, self.label2idx, self.attribute2value2idx = self.read_data()
        print(self._metadata_df.head())

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df["encoded_labels"])
        self._y_size = 1
        self._n_classes = len(self.label2idx)

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
                torch.LongTensor(self._metadata_df.loc[:, self._identity_vars].values),
                self._y_array.reshape((-1, 1)),
            ),
            dim=1,
        )
        self._metadata_fields = self._identity_vars + ["y"]

        self._eval_groupers = [
            CombinatorialGrouper(dataset=self, groupby_fields=self._groupby_fields)
        ]

        self._metric = F1(average="macro")

        super().__init__(root_dir, download, split_scheme)

    def read_data(self):

        train_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.train.jsonl"),
            data_type="train",
            attributes_to_retain=self._identity_vars,
        )
        dev_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.dev.jsonl"),
            data_type="dev",
            attributes_to_retain=self._identity_vars,
        )
        test_data = read_jsonl(
            os.path.join(self._data_dir, f"{self.dataset_name}.test.jsonl"),
            data_type="test",
            attributes_to_retain=self._identity_vars,
        )

        data = train_data + dev_data + test_data
        all_labels = sorted({example["labels"] for example in data})
        label2idx = dict(reversed(x) for x in enumerate(all_labels))
        attribute2value2idx = dict()
        for example in data:

            for a in self._identity_vars:
                v = example[a]
                a_vals = attribute2value2idx.get(a, set())
                a_vals.add(v)
                attribute2value2idx[a] = a_vals
        for a in self._identity_vars:
            a_vals = sorted(attribute2value2idx[a])
            v2i = dict(reversed(x) for x in enumerate(a_vals))
            del attribute2value2idx[a]
            attribute2value2idx[a] = v2i
        for example in data:
            example["encoded_labels"] = label2idx[example["labels"]]
            for a in self._identity_vars:
                example[a] = attribute2value2idx[a][example[a]]

        df = pd.DataFrame(data)
        # df = df.fillna("")
        return df, label2idx, attribute2value2idx

    def get_input(self, idx):
        return self._text_array[idx]

    def eval(self, y_pred, y_true, metadata):
        metric = F1(average="macro")
        eval_grouper = self._eval_groupers[0]

        return self.standard_group_eval(metric, eval_grouper, y_pred, y_true, metadata)

if __name__ == '__main__':
    ScotusDataset(download=True)