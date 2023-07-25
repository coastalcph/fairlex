import torch
import pandas as pd
from datasets import load_dataset
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import F1, multiclass_logits_to_pred
from wilds.common.grouper import CombinatorialGrouper


class SCOTUSDataset(WILDSDataset):
    """
    SCOTUS dataset.
    This is the 2021 SCOTUS dataset.

    Supported `split_scheme`:
        'official': official split

    Input (x):
        Case facts of maximum token length of 2048.

    Label (y):
        y is 1 if appeal is approved, otherwise 0

    Metadata:
        respondent_type: The type of respondent, which is a manual categorization (clustering) of respondents (defendants) in five categories (person, public entity, organization, facility and other)
        decision_direction: The direction of the decision, i.e., whether the decision is liberal, or conservative, provided by SCDB

    Website:
        https://github.com/coastalcph/fairlex
    """
    _dataset_name = 'fscs'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://zenodo.org/record/6322643/files/scotus.zip',
            'compressed_size': 4_066_541_568
        },
    }

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official', group_by_fields='respondent_type'):
        self._version = version
        # the official split is the only split
        self._split_scheme = split_scheme
        self._y_type = 'long'
        self._y_size = 2
        self._n_classes = 2
        # Load data
        self.data_df = self.read_dataset()
        print(self.data_df.head())

        # Get arrays
        self._input_array = list(self.data_df['text'])
        # Get metadata
        self._metadata_fields, self._metadata_array, self._metadata_map = self.load_metadata(self.data_df)
        # Get y from metadata
        self._y_array = torch.LongTensor(self.data_df['label'])
        # Set split info
        self.initialize_split_dicts()
        for split in self.split_dict:
            split_indices = self.data_df['data_type'] == split
            self.data_df.loc[split_indices, 'data_type'] = self.split_dict[split]
        self._split_array = self.data_df['data_type'].values
        # eval
        self.group_by_fields = group_by_fields
        self.initialize_eval_grouper()
        self._data_dir = 'data/datasets'
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._input_array[idx]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = F1(prediction_fn=multiclass_logits_to_pred, average='macro')
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

    def initialize_split_dicts(self):
        if self.split_scheme == 'official':
            self._split_dict = {'train': 0, 'val': 1, 'test': 2}
            self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')

    def load_metadata(self, data_df):
        # Get metadata
        columns = ['respondent_type', 'decision_direction', 'y']
        metadata_fields = ['respondent_type', 'decision_direction', 'y']
        metadata_df = data_df[columns].copy()
        metadata_df.columns = metadata_fields
        ordered_maps = {}
        ordered_maps['respondent_type'] = range(0, 6)
        ordered_maps['decision_direction'] = range(0, 2)
        ordered_maps['y'] = range(0, 2)
        metadata_map, metadata = map_to_id_array(metadata_df, ordered_maps)
        return metadata_fields, torch.from_numpy(metadata.astype('long')), metadata_map

    def initialize_eval_grouper(self):
        if self.split_scheme == 'official':
            self._eval_grouper = CombinatorialGrouper(
                dataset=self,
                groupby_fields=self.group_by_fields)
        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')

    def read_dataset(self):
        data = []
        for split in ['train', 'validation', 'test']:
            dataset = load_dataset('coastalcph/fairlex', 'scotus', split=split)
            for example in dataset:
                example['y'] = example['label']
                example['data_type'] = split if split != 'validation' else 'val'
                data.append(example)
        df = pd.DataFrame(data)
        df = df.fillna("")
        return df
