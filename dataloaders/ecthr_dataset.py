import torch
import pandas as pd
from datasets import load_dataset
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.utils import map_to_id_array
from configs.supported import F1, binary_logits_to_pred_v2
from wilds.common.grouper import CombinatorialGrouper

EAST_EUROPEAN_COUNTRIES = {'RUSSIA', 'TURKEY', 'UKRAINE', 'POLAND', 'BULGARIA', 'CROATIA', 'HUNGARY',
                           'ROMANIA', 'SLOVAKIA', 'MOLDOVA', 'SLOVENIA', 'LITHUANIA', 'SERBIA', 'AZERBAIJAN',
                           'CZECH REPUBLIC', 'GEORGIA', 'ESTONIA', 'BOSNIA & HERZEGOVINA', 'NORTH MACEDONIA',
                           'FORMER YUGOSLAV MACEDONIA', 'ARMENIA', 'LATVIA', 'MONTENEGRO'}

ECHR_ARTICLES = {
                 "2": "Right to life",
                 "3": "Prohibition of torture",
                 "5": "Right to liberty and security",
                 "6": "Right to a fair trial",
                 "8": "Right to respect for private and family life",
                 "9": "Freedom of thought, conscience and religion",
                 "10": "Freedom of expression",
                 "11": "Freedom of assembly and association",
                 "14": "Prohibition of discrimination",
                 "P1-1": "Protection of property",
                 "NV": "No Violation",
                 }

GENDERS = {'n/a': 0, 'male': 1, 'female': 2}
AGE_GROUPS = {'n/a': 0, '<=35': 1, '<=65': 2, '>65': 3}


class ECtHRDataset(WILDSDataset):
    """
    ECtHR dataset.
    This is a modified version of the 2021 ECtHR dataset.

    Supported `split_scheme`:
        'official': official split

    Input (x):
        Case facts of maximum token length of 2048.

    Label (y):
        y is a list of the ECHR article violations.

    Metadata:
        defendant_state: Defendant State group (C.E. European, Rest of Europe)
        applicant_gender: The gender of the applicant (N/A, Male, Female)
        applicant_age: The age group of the applicant (N/A, <=35, <=64, or older)

    Website:
        https://github.com/coastalcph/fairlex
    """
    _dataset_name = 'ecthr'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://zenodo.org/record/6322643/files/ecthr.zip',
            'compressed_size': 4_066_541_568
        },
    }

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official', group_by_fields='defendant_state'):
        self._version = version
        # the official split is the only split
        self._split_scheme = split_scheme
        self._y_type = 'long'
        self._y_size = len(ECHR_ARTICLES)
        self._n_classes = len(ECHR_ARTICLES)
        # Load data
        self.data_df = self.load_dataset()
        print(self.data_df.head())
        # Get arrays
        self._input_array = list(self.data_df['text'])
        # Get metadata
        self._metadata_fields, self._metadata_array, self._metadata_map = self.load_metadata(self.data_df)
        # Get y from metadata
        self._y_array = torch.FloatTensor(self.data_df['labels'])
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
        metric = F1(prediction_fn=binary_logits_to_pred_v2, average='macro')
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
        columns = ['defendant_state', 'applicant_age', 'applicant_gender']
        metadata_fields = ['defendant_state', 'applicant_age', 'applicant_gender']
        metadata_df = data_df[columns].copy()
        metadata_df.columns = metadata_fields
        ordered_maps = {}
        ordered_maps['defendant_state'] = range(0, 2)
        ordered_maps['applicant_gender'] = range(0, 3)
        ordered_maps['applicant_age'] = range(0, 4)
        metadata_map, metadata = map_to_id_array(metadata_df, ordered_maps)
        return metadata_fields, torch.from_numpy(metadata.astype('long')), metadata_map

    def initialize_eval_grouper(self):
        if self.split_scheme == 'official':
            self._eval_grouper = CombinatorialGrouper(
                dataset=self,
                groupby_fields=self.group_by_fields)
        else:
            raise ValueError(f'Split scheme {self.split_scheme} not recognized')

    def load_dataset(self):
        data = []
        for split in ['train', 'validation', 'test']:
            dataset = load_dataset('coastalcph/fairlex', 'ecthr', split=split)
            for example in dataset:
                # Convert labels to many-hot
                temp_labels = [0] * len(ECHR_ARTICLES)
                for label_idx in example['labels']:
                    temp_labels[label_idx] = 1
                # If no label (no violation), set to last index
                if sum(temp_labels) == 0:
                    temp_labels[-1] = 1
                example['y'] = example['labels'] if len(example['labels']) else [dataset.features['labels'].feature.num_classes]
                example['labels'] = temp_labels
                example['data_type'] = split if split != 'validation' else 'val'
                data.append(example)
        df = pd.DataFrame(data)
        df = df.fillna("")
        return df
