import pandas as pd
import numpy as np
# from dataloaders import get_dataset
from sklearn.metrics import f1_score
from scipy.special import expit
import os
# DATASET = 'ecthr'
DATASET = 'fscs'
# GROUP_FIELDS = {'gender': (1, 3), 'age': (1, 4), 'defendant': (0, 2)}
GROUP_FIELDS = {'language': (0, 3), 'region': (1, 9), 'legal_area': (1, 6)}
LOG_DIR = 'linear_logs/batch_12'  # 'linear_logs/batch_12'

for group_field, (first_group, no_groups) in GROUP_FIELDS.items():
    if not os.path.exists(f'{LOG_DIR}/{DATASET}/ERM/{group_field}'):
        continue
    print('-' * 150)
    print(f'{group_field.upper()} ({no_groups} GROUPS)')
    print('-' * 150)
    # dataset = get_dataset(DATASET, group_by_fields=[group_field], root_dir='../data/datasets')
    for algorithm in ['ERM', 'ERM-GS', 'IRM', 'groupDRO', 'REx']:
        if not os.path.exists(f'{LOG_DIR}/{DATASET}/{algorithm}/{group_field}'):
            continue

        scores = {'val': {'mF1': [], 'mF1[group]': [], 'GD': []},
        'test': {'mF1': [], 'mF1[group]': [], 'GD': []}}

        for group_no in range(first_group, no_groups):
            scores['val'].update({f'mF1 ({group_no+1})': []})
        for group_no in range(first_group, no_groups):
            scores['test'].update({f'mF1 ({group_no+1})': []})

        # scores['val'].update({f'Micro-F1': []})
        # scores['test'].update({f'Micro-F1': []})
        #
        # for group_no in range(no_groups):
        #     scores['val'].update({f'Micro-F1 ({group_no + 1})': []})
        # for group_no in range(no_groups):
        #     scores['test'].update({f'Micro-F1 ({group_no + 1})': []})

        try:
            for seed_no in range(1, 5):
                try:
                    for split in ['val', 'test']:
                        # ORIGINAL SCORES
                        original_df = pd.read_csv(f'{LOG_DIR}/{DATASET}/{algorithm}/{group_field}/seed_{seed_no}/{split}_eval.csv')
                        scores[split]['mF1'].append(original_df['F1-macro_all'].values[-1])
                        group_wise_scores = []
                        for group_no in range(first_group, no_groups):
                            scores[split][f'mF1 ({group_no+1})'].append(original_df[f'F1-macro_{group_field}:{group_no}'].values[-1])
                            group_wise_scores.append(original_df[f'F1-macro_{group_field}:{group_no}'].values[-1])
                        group_wise_scores = [s for s in group_wise_scores if s]
                        scores[split]['GD'].append(np.std(group_wise_scores))
                        scores[split]['mF1[group]'].append(group_wise_scores)
                        # scores[split]['mF1[worst]'].append(min(group_wise_scores))

                        # # RE-COMPUTED SCORES
                        # y_true = dataset.get_subset(f'{split}').y_array.numpy()
                        # y_pred = pd.read_csv(
                        #     f'../logs_final/{DATASET}/{algorithm}/{group_field}/seed_{seed_no}/ecthr_split:{split}_seed:{seed_no}_epoch:best_pred.csv',
                        #     header=None).values
                        # y_pred = (expit(y_pred) > 0.5).astype('int')
                        # groups = dataset.get_subset(f'{split}').metadata_array[:, 2].numpy()
                        # scores[split]['Micro-F1'].append(f1_score(y_true, y_pred, average='micro'))
                        # for group_no in range(no_groups):
                        #     y_pred_g = []
                        #     y_true_g = []
                        #     for y, y_hat, group in zip(y_true, y_pred, groups):
                        #         if group == group_no:
                        #             y_true_g.append(y)
                        #             y_pred_g.append(y_hat)
                        #     scores[split][f'Micro-F1 ({group_no+1})'].append(f1_score(y_true_g, y_pred_g, average='micro', zero_division=0))

                except:
                    continue

            print('-' * 150)
            print(f'{algorithm.upper()} ({algorithm.upper()})')
            print('-' * 150)
            for split in ['val', 'test']:
                print(f'{split.upper()}:\t' + '\t'.join([f'{k}: {np.mean(v):.2%} ± {np.std(v):.2%}' for k, v in scores[split].items()]))
            print()
        except:
            continue

