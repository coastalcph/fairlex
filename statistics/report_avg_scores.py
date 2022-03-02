import pandas as pd
import numpy as np
import os

LOG_DIR = 'logs'
DATASET = 'ecthr'
GROUP_FIELDS = {'applicant_gender': (1, 3), 'applicant_age': (1, 4), 'defendant_state': (0, 2)}

for group_field, (first_group, no_groups) in GROUP_FIELDS.items():
    if not os.path.exists(f'{LOG_DIR}/{DATASET}/ERM/{group_field}'):
        continue
    print('-' * 150)
    print(f'{group_field.upper()} ({no_groups} GROUPS)')
    print('-' * 150)
    for algorithm in ['ERM', 'ERM-GS', 'adversarialRemoval', 'IRM', 'groupDRO', 'REx']:
        if not os.path.exists(f'{LOG_DIR}/{DATASET}/{algorithm}/{group_field}'):
            continue

        scores = {'val': {'mF1': [], 'mF1[group]': [], 'GD': []},
        'test': {'mF1': [], 'mF1[group]': [], 'GD': []}}

        for group_no in range(first_group, no_groups):
            scores['val'].update({f'mF1 ({group_no+1})': []})
        for group_no in range(first_group, no_groups):
            scores['test'].update({f'mF1 ({group_no+1})': []})
        try:
            for seed_no in range(1, 4):
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

