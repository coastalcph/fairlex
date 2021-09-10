import pandas as pd

ALGORITHM = 'REx'
DATASET = 'ecthr'
GROUP_FIELD = 'age'
N_GROUPS = 3

scores = {'val': {'Macro-F1': []}.update({f'Macro-F1: ({group_no})': [] for group_no in range(N_GROUPS)}),
          'test': {'Macro-F1': []}.update({f'Macro-F1: ({group_no})': [] for group_no in range(N_GROUPS)})}

for seed_no in range(1, 6):
    for split in ['val', 'test']:
        df = pd.read_csv(f'../logs/{DATASET}/{ALGORITHM}/{GROUP_FIELD}/seed_{seed_no}/{split}_eval.csv')
        scores[split]['Macro-F1'].append(df['F1-macro_all'].values[-1])
        scores[split]['Macro-F1 (0)'].append(df['F1-macro_age:0'].values[-1])
        scores[split]['Macro-F1 (1)'].append(df['F1-macro_age:1'].values[-1])
        scores[split]['Macro-F1 (2)'].append(df['F1-macro_age:2'].values[-1])

for split in ['val', 'test']:
    print('\t'.join([f'{k}: {v:%.2f}' for k, v in scores[split]]))

