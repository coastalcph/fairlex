from dataloaders.scotus_attribute_mapping import ISSUE_AREA_MAPPING, ISSUE_AREAS
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math
def __load_model(path):
    state = torch.load(path, map_location='cpu')
    model = state['algorithm']
    weights = model['model.classifier.weight']
    bias = model['model.classifier.bias']
    return weights.numpy(), bias.numpy()

def get_most_important_words_by_label(text_vectorizer:TfidfVectorizer, label_map, model_path, k):
    if k is None: 
        k = 0
    weights, bias = __load_model(model_path)
    top_100_idxs_by_class = np.argsort(weights, -1)[:, -k:]
    top_100_scores_by_class = np.sort(weights, -1)[:, -k:]
    features = text_vectorizer.get_feature_names()
    top_100_features_by_class = [[features[i] for i in fs] for fs in top_100_idxs_by_class]
    top_100 = dict()

    for i, (fs, fscores) in enumerate(zip(top_100_features_by_class, top_100_scores_by_class)):
        label = label_map[i]
        top_100[label] = list(zip(fs, fscores))
    return top_100
    
    
def __dict_diff(a, b):
    new_a = dict()
    for k,a_v in a.items():
        b_v = {i:ii for i, ii in b[k]}
        a_v = {i:ii for i, ii in a_v}
        new_a[k] = list()
        for kk, vv in a_v.items():
            bvv = b_v[kk]
            new_a[k].append((kk, vv-bvv))

    for k, vals in new_a.items():
        new_a[k] = sorted(vals, key=lambda x : x[1])
    return new_a
    

    
import pickle as pkl
from tabulate import tabulate
from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--biased_model_path', required=True, type=str)
    parser.add_argument('--original_model_path', required=True, type=str)
    parser.add_argument('--tokenizer_path', required=True, type=str)
    parser.add_argument('--latex', default=False)
    args = parser.parse_args()

    latex = args.latex
    with open(args.tokenizer_path, 'rb') as lines:
        vectorizer = pkl.load(lines)
    label_map = ISSUE_AREAS
    
    biased_model_path=args.biased_model_path
    original_model_path=args.original_model_path
    biased_ranking = get_most_important_words_by_label(vectorizer, label_map, biased_model_path, None)
    original_ranking = get_most_important_words_by_label(vectorizer, label_map, original_model_path, None)
    diff_ranking = __dict_diff(biased_ranking, original_ranking)
    
    data_frames = dict()
    for label, diff_words in diff_ranking.items():
        print('%' * 150)
        print('%' + label)
        print('%' * 150)
        d = {'original':list(), 'original_scores':list(), "biased":list(), 'biased_scores':list(), 'peculiar':list(), 'peculiar_scores':list()}
        original_words = sorted(original_ranking[label][-50:] + original_ranking[label][:50], key=lambda x: -x[1])
        biased_words = sorted(biased_ranking[label][-50:] + biased_ranking[label][:50], key=lambda x : -x[1])
        diff_words = sorted(diff_words[-50:] + diff_words[:50], key=lambda x: -x[1])
        for (t, s), (ot, os), (bt, bs) in list(zip(diff_words, original_words, biased_words))[:10] + list(zip(diff_words, original_words, biased_words))[-10:]:
            d['original'].append(ot)
            d['original_scores'].append(f"{os:.02f}")
            d['biased'].append(bt)
            d['biased_scores'].append(f"{bs:.02f}")
            d['peculiar'].append(t)
            d['peculiar_scores'].append(f"{s:.02f}")
        df = pd.DataFrame(data=d)
        if latex:
            print(df.to_latex(index=False))
        else:
            print(tabulate(df, headers='keys', tablefmt='psql'))
