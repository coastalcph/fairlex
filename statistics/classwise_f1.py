import pandas as pd
from argparse import ArgumentParser
import numpy as np
from dataloaders import get_dataset
from sklearn.metrics import f1_score
from scipy.special import expit, softmax
from pprint import pprint
import os
import logging
logging.getLogger("pandas").setLevel(logging.ERROR)
DATASET = 'scotus'
ECTHR_GROUP_FIELDS = {'gender': (1, 3), 'age': (1, 4), 'defendant': (0, 2)}
FSCS_GROUP_FIELDS = {'language': (0, 3), 'region': (1, 9), 'legal_area': (1, 6)}
SPC_GROUP_FIELDS = {'gender': (0, 2), 'region': (0, 7)}
SCOTUS_GROUP_FIELDS =  {'respondent': (0,5), 'decisionDirection': (0, 2)}
ALL_GROUP_FIELDS={'ecthr':ECTHR_GROUP_FIELDS, 'fscs':FSCS_GROUP_FIELDS, 'spc':SPC_GROUP_FIELDS, 'scotus':SCOTUS_GROUP_FIELDS}

def add_zero_class(labels):
    augmented_labels = np.zeros((len(labels), len(labels[0]) + 1), dtype=np.int32)
    augmented_labels[:, :-1] = labels
    augmented_labels[:, -1] = (np.sum(labels, axis=1) == 0).astype('int32')
    return augmented_labels

def get_classwise_f1(dataset_name):
    LOG_DIR = 'logs_final_mini_roberta/'  # 'linear_logs/batch_12'
    #first_attribute = list(GROUP_FIELDS.keys())[0]
    #dataset = get_dataset(DATASET, group_by_fields=[first_attribute])
    GROUP_FIELDS = ALL_GROUP_FIELDS[dataset_name]
    algo2groupfield2attval2score=dict()
    for algorithm in ['ERM', 'adversarialRemoval', 'groupDRO', 'IRM', 'REx']:
        groupfield2attval2score = dict()
        for group_field, (first_group, no_groups) in GROUP_FIELDS.items():
            log_path = f'{LOG_DIR}/{dataset_name}/{algorithm}/{group_field}'
            if not os.path.exists(log_path):
                print(f'cannot find {log_path}')
                continue
            dataset = get_dataset(dataset_name, group_by_fields=[group_field])
            scores = {group_no: [] for group_no in range(first_group, no_groups)}
            examples = {group_no: [] for group_no in range(first_group, no_groups)}
            scores['X'] = []
            examples['X'] = []
            for seed_no in range(1, 4):
                for split in ['test']:
                    # RE-COMPUTED SCORES
                    subset = dataset.data_df[dataset.data_df['data_type'] == dataset.split_dict[split]]
                    log_path = f'{LOG_DIR}/{dataset_name}/{algorithm}/{group_field}/seed_{seed_no}/{dataset_name}_split:{split}_seed:{seed_no}_epoch:best_pred.csv'
                    if not os.path.exists(log_path):
                        continue
                    y_log = pd.read_csv(
                        log_path,
                        header=None).values
                    if dataset_name == 'fscs':
                        y_pred = [[0, 1] if val else [1, 0] for val in np.argmax(y_log,  axis=-1)]
                        labels = [[0, 1] if val else [1, 0] for val in list(subset['y'])]
                    elif dataset_name  == 'spc':
                        y_idx = np.argmax(y_log,  axis=-1)
                        y_pred = [[0] * len(y_log[0]) for _ in y_log]
                        for idx, i in enumerate(y_idx):
                            y_pred[idx][i] = 1
                        y_true = list(subset['label'])
                        labels = [[0] * len(y_pred[0]) for _ in y_pred]
                        for idx, i in enumerate(y_true):
                            labels[idx][i] = 1
                    elif dataset_name == 'scotus':
                        y_true = list(subset['label_id'])
                        num_labels = max(y_true) + 1
                        y_idx = np.argmax(y_log,  axis=-1)
                        y_pred = [[0] * num_labels for _ in y_log]
                        for idx, i in enumerate(y_idx):
                            y_pred[idx][i] = 1
                        labels = [[0] * num_labels for _ in y_pred]
                        for idx, i in enumerate(y_true):
                            labels[idx][i] = 1
                    else:
                        y_pred = (expit(y_log) > 0.5).astype('int')
                        y_true = list(subset['labels'])
                        y_pred = add_zero_class(y_pred)
                        labels = add_zero_class(y_true)

                    subset['labels'] = list(labels)
                    subset['predictions'] = list(y_pred)
                    group_y_true = []
                    group_y_pred = []
                    for group_no in range(first_group, no_groups):
                        group_dataset = subset[subset[group_field] == group_no]
                        group_y_true += group_dataset.labels.tolist()
                        group_y_pred += group_dataset.predictions.tolist()
                    scores['X'].append(f1_score(group_y_true, group_y_pred, average=None))
                    examples['X'] = np.sum(group_y_true, axis=0)
                    for group_no in range(first_group, no_groups):
                        group_dataset = subset[subset[group_field] == group_no]
                        group_y_true = group_dataset.labels.tolist()
                        group_y_pred = group_dataset.predictions.tolist()
                        f1 = f1_score(group_y_true, group_y_pred, average=None)
                        scores[group_no].append(f1)
                        examples[group_no] = np.sum(group_y_true, axis=0)
            temp_scores = [f'{val * 100:.1f}' for val in np.mean(scores['X'], axis=0)]
            micro_f1 = sum([float(score) * num for score, num in zip(temp_scores, examples['X'])]) / np.sum(examples['X'])
            att_val2score = dict()
            for group_no in range(first_group, no_groups):
                temp_scores = [f'{val*100:.1f}' for val in np.mean(scores[group_no], axis=0)]
                if len(temp_scores) == 0:
                    print(f'no scores for attribute value {group_no} of group {group_field}')
                    continue
                micro_f1 = sum([float(score) * num for score, num in zip(temp_scores, examples[group_no])]) / np.sum(examples[group_no])
                att_val2score[group_no] = np.mean([float(val) for val in temp_scores])
            groupfield2attval2score[group_field] = att_val2score
        algo2groupfield2attval2score[algorithm] = groupfield2attval2score

    return algo2groupfield2attval2score 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    pprint(get_classwise_f1(args.dataset))
