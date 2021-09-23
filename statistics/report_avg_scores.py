import pandas as pd
import numpy as np
from dataloaders import get_dataset
from sklearn.metrics import f1_score
from scipy.special import expit
import os
DATASET = 'scotus'
GROUP_FIELD = 'age'
N_GROUPS = 4
SPLIT_SCHEME = 'official'

dataset = get_dataset(DATASET, group_by_fields=[GROUP_FIELD], split_scheme='official', protected_attribute=GROUP_FIELD, root_dir='data/datasets')
for algorithm in ['ERM', 'groupDRO', 'IRM']:
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

    for seed_no in range(1, 5):
        log_path = f'../logs_final/{DATASET}/{algorithm}/{GROUP_FIELD}/seed_{seed_no}/'
    
        for split in ['val', 'test']:
            # ORIGINAL SCORES
            original_df = pd.read_csv(os.path.join(log_path, f'{split}_eval.csv'))
            scores[split]['Macro-F1'].append(original_df['F1-macro_all'].values[-1])
            for group_no in range(N_GROUPS):
                scores[split][f'Macro-F1 ({group_no+1})'].append(original_df[f'F1-macro_{GROUP_FIELD}:{group_no}'].values[-1])

            # RE-COMPUTED SCORES
            y_true = dataset.get_subset(f'{split}').y_array.numpy()
            y_pred = pd.read_csv(
                os.path.join(log_path, f'{DATASET}_split:{split}_seed:{seed_no}_epoch:best_pred.csv'),
                header=None).values
            y_pred = (expit(y_pred) > 0.5).astype('int')
            groups = dataset.get_subset(f'{split}').metadata_array[:, 2].numpy()
            scores[split]['Micro-F1'].append(f1_score(y_true, y_pred, average='micro'))
            for group_no in range(3):
                y_pred_g = []
                y_true_g = []
                for y, y_hat, group in zip(y_true, y_pred, groups):
                    if group == group_no:
                        y_true_g.append(y)
                        y_pred_g.append(y_hat)
                scores[split][f'Micro-F1 ({group_no+1})'].append(f1_score(y_true_g, y_pred_g, average='micro'))

    print('-' * 150)
    print(f'{algorithm.upper()}')
    print('-' * 150)
    for split in ['val', 'test']:
        print(f'{split.upper()}:\t' + '\t'.join([f'{k}: {np.mean(v):.2%} ± {np.std(v):.2%}' for k, v in scores[split].items()]))

