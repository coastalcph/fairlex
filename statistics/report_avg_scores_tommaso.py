import pandas as pd
import numpy as np
# from dataloaders import get_dataset
from sklearn.metrics import f1_score
from scipy.special import expit
import os
from argparse import ArgumentParser
# DATASET = 'ecthr'
# DATASET = 'fscs'
# GROUP_FIELDS = {'gender': (1, 3), 'age': (1, 4), 'defendant': (0, 2)}
# GROUP_FIELDS = {'language': (0, 3), 'region': (1, 9), 'legal_area': (1, 6)}
# LOG_DIR = 'linear_logs/batch_12'  # 'linear_logs/batch_12'

GROUP_FIELDS_BY_DATASET={'scotus':{'decisionDirection': (0,2), 'respondent':(0, 5)}, 'ecthr':{'gender': (1, 3), 'age': (1, 4), 'defendant': (0, 2)}, 
'fscr': {'language': (0, 3), 'region': (1, 9), 'legal_area': (1, 6)}}

def compute(dataset, log_dir):
    group_fields = GROUP_FIELDS_BY_DATASET[dataset]
    for group_field, (first_group, no_groups) in group_fields.items():
        if not os.path.exists(f'{log_dir}/{dataset}/ERM/{group_field}'):
            print('skipping', f'{log_dir}/{dataset}/ERM/{group_field}')
            continue
        print('-' * 150)
        print(f'{group_field.upper()} ({no_groups} GROUPS)')
        print('-' * 150)
        # dataset = get_dataset(DATASET, group_by_fields=[group_field], root_dir='../data/datasets')
        for algorithm in ['ERM', 'adversarialRemoval', 'groupDRO', 'IRM', 'REx']:
            if not os.path.exists(f'{log_dir}/{dataset}/{algorithm}/{group_field}'):
                continue

            scores = {'val': {'mF1': [], 'mF1[group]': [], 'GD': []},
            'test': {'mF1': [], 'mF1[group]': [], 'GD': []}}

            for group_no in range(first_group, no_groups):
                scores['val'].update({f'mF1 ({group_no+1})': []})
            for group_no in range(first_group, no_groups):
                scores['test'].update({f'mF1 ({group_no+1})': []})


            try:
                for seed_no in range(1, 5):
                    try:
                        for split in ['val', 'test']:
                            # ORIGINAL SCORES
                            original_df = pd.read_csv(f'{log_dir}/{dataset}/{algorithm}/{group_field}/seed_{seed_no}/{split}_eval.csv')
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['scotus', 'ecthr', 'fscs'], required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    args = parser.parse_args()
    compute(args.dataset, args.log_dir)

