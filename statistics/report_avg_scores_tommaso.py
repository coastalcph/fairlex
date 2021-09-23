import pandas as pd
import numpy as np
from dataloaders import get_dataset
from sklearn.metrics import f1_score
import os
from scipy.special import expit




n_groups_by_field = {'respondent': 5, 'decisionDirection': 2, 'age':4, 'gender':3, 'defendant':2}

DATASET = 'scotus'
GROUP_FIELD = 'decisionDirection'
N_GROUPS = n_groups_by_field[GROUP_FIELD]
SPLIT_SCHEME = 'official'
logs_path_base = f'logs_final/{DATASET}/'
available_algorithms = ['ERM', 'adversarialRemoval', 'groupDRO', 'IRM', 'REx']
dataset = get_dataset(DATASET, group_by_fields=[GROUP_FIELD], split_scheme='official', root_dir='data/datasets')

for algorithm in available_algorithms:
    algo_log_path = os.path.join(logs_path_base, algorithm, GROUP_FIELD)
    scores = {'val': {'Macro-F1': []},
              'test': {'Macro-F1': []}}

    for group_no in range(N_GROUPS):
        scores['val'].update({f'Macro-F1 ({group_no+1})': []})
    for group_no in range(N_GROUPS):
        scores['test'].update({f'Macro-F1 ({group_no+1})': []})

    scores['val'].update({f'Micro-F1': []})
    scores['test'].update({f'Micro-F1': []})

    for group_no in range(N_GROUPS):
        scores['val'].update({f'Micro-F1 ({group_no + 1})': []})
    for group_no in range(N_GROUPS):
        scores['test'].update({f'Micro-F1 ({group_no + 1})': []})
    metrics_to_compute = 'F1-micro'
    metrics_available = 'F1-macro'
    metrics_to_compute_name = 'Micro-F1'
    metrics_available_name = 'Macro-F1'
    for seed_no in range(1, 6):
        seed_log_path = os.path.join(algo_log_path, f'seed_{seed_no}')
    
        for split in ['val', 'test']:
            # ORIGINAL SCORES
            original_df = pd.read_csv(os.path.join(seed_log_path, f'{split}_eval.csv'))
            if any(metrics_available in x for x in original_df.columns):
                metrics_to_compute = 'F1-macro'
                metrics_to_compute_name = 'Macro-F1'
            else:
                assert any(metrics_to_compute in x for x in original_df.columns)
                aux = metrics_to_compute
                aux_name = metrics_to_compute_name
                metrics_to_compute = metrics_available
                metrics_to_compute_name = metrics_available_name
                metrics_available = aux
                metrics_available_name = aux_name
            scores[split][metrics_available_name].append(original_df[metrics_available + '_all'].values[-1])
            for group_no in range(N_GROUPS):
                scores[split][f'{metrics_available_name} ({group_no+1})'].append(original_df[f'{metrics_available}_{GROUP_FIELD}:{group_no}'].values[-1])

            # RE-COMPUTED SCORES
            y_true = dataset.get_subset(f'{split}').y_array.numpy()
            y_pred = pd.read_csv(
                os.path.join(seed_log_path, f'{DATASET}_split:{split}_seed:{seed_no}_epoch:best_pred.csv'),
                header=None).values
            labels = None
            if DATASET == 'scotus':
                labels = list(range(len(y_true[0])))
                y_pred = np.argmax(y_pred, -1).astype('int')
                y_true = np.argmax(y_true, -1).astype('int')
            else:
                y_pred = (expit(y_pred) > 0.5).astype('int')

            group_position = dataset._metadata_fields.index(GROUP_FIELD)
            groups = dataset.get_subset(f'{split}').metadata_array[:, group_position].numpy()
            scores[split][metrics_to_compute_name].append(f1_score(y_true, y_pred, average='macro' if 'macro' in metrics_to_compute else 'micro', labels=labels, zero_division=0))
            for group_no in range(N_GROUPS):
                y_pred_g = []
                y_true_g = []
                for y, y_hat, group in zip(y_true, y_pred, groups):
                    if group == group_no:
                        y_true_g.append(y)
                        y_pred_g.append(y_hat)
                    
                score = f1_score(y_true_g, y_pred_g, average='macro' if 'macro' in metrics_to_compute else 'micro', labels = labels, zero_division=0)
                scores[split][f'{metrics_to_compute_name} ({group_no+1})'].append(score)
    print('-' * 150)
    print(f'{algorithm.upper()}')
    print('-' * 150)
    for split in ['val', 'test']:
        print(f'{split.upper()}:\t' + '\t'.join([f'{k}: {np.mean(v):.2%} ± {np.std(v):.2%} {sorted(v)[0]:.2%}' for k, v in scores[split].items() if 'Macro' in k] ))

