from distribution_shift import compute_distribution_divergence, jsd
from argparse import ArgumentParser
import os
from pprint import pprint
import pickle as pkl
from scipy.stats import pearsonr
from argparse import ArgumentParser
from classwise_f1 import get_classwise_f1
import numpy as np

ADV_REMOVAL_F1s = {'ecthr':{'defendant':53.8, 'gender':54.6, 'age':48.9}, 'scotus':{'respondent':56.9, 'decisionDirection':41.0}, 'fscs':{'language':62.6, 'legal_area':65.6, 'region':67.4}, 'spc':{'gender':82.6, 'region':83.4}}
GDRO_F1s = {'ecthr':{'defendant':55.0, 'gender':56.3, 'age':52.6}, 'scotus':{'respondent':74.5, 'decisionDirection':77.1}, 'fscs':{'language':70.5, 'legal_area':65.6, 'region':67.4}, 'spc':{'gender':83.5, 'region':82.5}}
IRM_F1s = {'ecthr':{'defendant':53.8, 'gender':53.8, 'age':54.8}, 'scotus':{'respondent':73.4, 'decisionDirection':78.1}, 'fscs':{'language':68.3, 'legal_area':67.8, 'region':68.7}, 'spc':{'gender':83.8, 'region':83.5}}
REX_F1s = {'ecthr':{'defendant':54.6, 'gender':54.6, 'age':55.0}, 'scotus':{'respondent':73.8, 'decisionDirection':78.2}, 'fscs':{'language':67.8, 'legal_area':69.4, 'region':69.7}, 'spc':{'gender':83.8, 'region':83.1}}
ERM_F1s = {'ecthr':{'defendant':53.2, 'gender':57.5, 'age':54.1}, 'scotus':{'respondent':75.2, 'decisionDirection':77.1}, 'fscs':{'language':67.2, 'legal_area':66.6, 'region':68.4}, 'spc':{'gender':84.8, 'region':83.5}}
ALGO_F1s = {'ERM':ERM_F1s, 'adversarialRemoval': ADV_REMOVAL_F1s, 'groupDRO': GDRO_F1s, 'IRM': IRM_F1s, 'REx': REX_F1s}

def min_max_norm(serie):
    values = np.array([x for _, x in serie])
    indices = [i for i, _ in serie]
    _min = min(values)
    _max = max(values)
    values = (values - _min) /  (_max - _min)
    return list(zip(indices, values.tolist()))
if __name__ == '__main__':
    parser = ArgumentParser()
    algo2correlation = dict()
    for algo in ['ERM']: #ALGO_F1s.keys():
        all_norm_f1s = []
        all_norm_divs = []
        for dataset in ['ecthr', 'scotus', 'fscs', 'spc']:
            algo2group2attval2score = get_classwise_f1(dataset) 
            F1s = ALGO_F1s[algo]
            group2attval2score = algo2group2attval2score[algo]
            attributes = F1s[dataset].keys()
            for p_attribute in attributes:
                divergences = []
                f1s = []
                path = f'.{dataset}.{p_attribute}.pkl'
                attval2score = group2attval2score[p_attribute]
                if os.path.exists(path):
                    with open(path, 'rb') as reader:
                        distribution_divergence = pkl.load(reader)
                else:
                    distribution_divergence = compute_distribution_divergence(dataset, jsd, p_attribute)['train-test divergence']
                    with open(path, 'wb') as writer:
                        pkl.dump(distribution_divergence, writer)
                #print(distribution_divergence)
                #print(attval2score)
                for div, idx in distribution_divergence:
                    if idx not in attval2score:
                        continue
                    f1 = attval2score[idx]
                    f1s.append((idx,f1))
                    divergences.append((idx, div))
            f1s_norm = min_max_norm(f1s)
            divs_norm = min_max_norm(divergences)
            all_norm_f1s.extend(f1s_norm)
            all_norm_divs.extend(divs_norm)

        pr = pearsonr([d for _, d in all_norm_divs], [f1 for _, f1, in all_norm_f1s])
        #print('F1s=', all_norm_f1s)
        #print('Divs=', all_norm_divs)
        print(f'pearson g= {pr[0]}\np = {pr[1]}')
        algo2correlation[algo] = pr
    pprint(algo2correlation)

