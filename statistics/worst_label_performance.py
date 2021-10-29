import pandas as pd
import numpy as np
from dataloaders import get_dataset
from sklearn.metrics import f1_score
from scipy.special import expit, softmax
import os
import logging
logging.getLogger("pandas").setLevel(logging.ERROR)
# DATASET = 'ecthr'
#DATASET = 'fscs'
DATASET = 'scotus'
# DATASET = 'spc'
# GROUP_FIELDS = {'gender': (1, 3), 'age': (1, 4), 'defendant': (0, 2)}
# GROUP_FIELDS = {'language': (0, 3), 'region': (1, 9), 'legal_area': (1, 6)}
# GROUP_FIELDS = {'gender': (0, 2), 'region': (0, 7)}
GROUP_FIELDS = {'decisionDirection':(0,2), 'respondent':(0,6)}

def add_zero_class(labels):
    augmented_labels = np.zeros((len(labels), len(labels[0]) + 1), dtype=np.int32)
    augmented_labels[:, :-1] = labels
    augmented_labels[:, -1] = (np.sum(labels, axis=1) == 0).astype('int32')
    return augmented_labels


LOG_DIR = 'logs_final_mini_roberta/'  # 'linear_logs/batch_12'
first_attribute = list(GROUP_FIELDS.keys())[0]
for group_field, (first_group, no_groups) in GROUP_FIELDS.items():
    if not os.path.exists(f'{LOG_DIR}/{DATASET}/ERM/{group_field}'):
        continue
    dataset = get_dataset(DATASET, group_by_fields=[group_field])
    print('-' * 150)
    print(f'{group_field.upper()} ({no_groups} GROUPS)')
    print('-' * 150)
    for algorithm in ['ERM']:
        if not os.path.exists(f'{LOG_DIR}/{DATASET}/{algorithm}/{group_field}'):
            continue
        scores = {group_no: [] for group_no in range(first_group, no_groups)}
        examples = {group_no: [] for group_no in range(first_group, no_groups)}
        scores['X'] = []
        examples['X'] = []
        for seed_no in range(1, 4):
            for split in ['test']:
                # RE-COMPUTED SCORES
                subset = dataset.data_df[dataset.data_df['data_type'] == dataset.split_dict[split]]
                y_log = pd.read_csv(
                    f'{LOG_DIR}/{DATASET}/{algorithm}/{group_field}/seed_{seed_no}/{DATASET}_split:{split}_seed:{seed_no}_epoch:best_pred.csv',
                    header=None).values

                if DATASET == 'fscs':
                    y_pred = [[0, 1] if val else [1, 0] for val in np.argmax(y_log,  axis=-1)]
                    labels = [[0, 1] if val else [1, 0] for val in list(subset['y'])]
                elif DATASET == 'spc':
                    y_idx = np.argmax(y_log,  axis=-1)
                    y_pred = [[0] * len(y_log[0]) for _ in y_log]
                    for idx, i in enumerate(y_idx):
                        y_pred[idx][i] = 1
                    y_true = list(subset['y'])
                    labels = [[0] * len(y_pred[0]) for _ in y_pred]
                    for idx, i in enumerate(y_true):
                        labels[idx][i] = 1
                elif DATASET == 'scotus':
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
                    y_true = list(subset['y'])
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
    print(f'Labels : ' + '\t'.join([f'{idx:>13} |' for idx, _ in enumerate(temp_scores)]) + '| μ-F1 | m-F1 Total')
    print('-'*200)
    print(f'Group X: ' + '\t'.join([f'{score:>5} ({num:>5}) |' for score, num in zip(temp_scores, examples['X'])]) + f'| {np.mean([float(val) for val in temp_scores]):.1f} | {micro_f1:.1f} ({np.sum(examples["X"]):>5})')
    for group_no in range(first_group, no_groups):
        temp_scores = [f'{val*100:.1f}' for val in np.mean(scores[group_no], axis=0)]
        micro_f1 = sum([float(score) * num for score, num in zip(temp_scores, examples[group_no])]) / np.sum(examples[group_no])
        print(f'Group {group_no}: ' + '\t'.join([f'{score:>5} ({num:>5}) |' for score, num in zip(temp_scores, examples[group_no])]) + f'| {np.mean([float(val) for val in temp_scores]):.1f} | {micro_f1:.1f} ({np.sum(examples[group_no]):>5})')
