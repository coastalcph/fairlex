from collections import defaultdict
import numpy as np
from numpy.lib.function_base import average
from sklearn.feature_extraction.text import TfidfVectorizer
from dataloaders import get_dataset
import os
import torch
import pickle as pkl
import pandas as pd
from tabulate import tabulate
from argparse import ArgumentParser

def __load_model(path):
    state = torch.load(path, map_location='cpu')
    model = state['algorithm']
    weights = model['model.classifier.weight']
    bias = model['model.classifier.bias']
    return weights.numpy(), bias.numpy()


def load_answers(path, answer_processor_fun):
    answers = list()
    with open(path) as lines:
        for line in lines:
            answers.append(answer_processor_fun(line.strip()))
    return np.stack(answers, 0)

def average_features(dataset, logdir, vectorizer_path, attribute, attribute_val, attribute_val2idx, weight_by_param=False, params=None, k = 20):
    with open(vectorizer_path, 'rb') as reader:
        vectorizer:TfidfVectorizer = pkl.load(reader)

    valset = dataset.get_subset('val')
    answers_path = os.path.join(logdir, f'{dataset_name}_split:val_seed:1_epoch:best_pred.csv')
    if 'scotus' == dataset_name or 'ecthr' == dataset_name:
        answer_processor_fun = lambda line : np.argmax(np.array([float(x) for x in line.split(',')]), -1)
    else:
        raise RuntimeError('Not supported yet')

    answers = load_answers(answers_path, answer_processor_fun)
    gold_answers = np.argmax(valset.y_array, -1).numpy()
    wrong_answer_mask = gold_answers != answers
    print(attribute_val2idx)
    attribute_val_idx = attribute_val2idx[attribute_val]
    attribute_mask = valset.metadata_array[:,valset.metadata_fields.index(attribute)].numpy() == attribute_val_idx
    #attribute_mask = np.array([1 if a == attribute_val_idx else 0 for _, _, a in valset])
    wrong_answer_mask = wrong_answer_mask * attribute_mask
    wrong_answer_indices = np.where(wrong_answer_mask)[0]
    wrong_predictions = [answers[i] for i in wrong_answer_indices]
    wrong_examples = [valset[i] for i in wrong_answer_indices]
    wrong_examples_features = np.array(vectorizer.transform([example[0] for example in wrong_examples]).todense())
    if weight_by_param and params is not None:
        param_weights = np.stack([params[p] for p in wrong_predictions], 0)
        wrong_examples_features = np.multiply(wrong_examples_features, param_weights)
    mean_features_scores = np.mean(wrong_examples_features, 0)

    mean_sorted_indices = np.argsort(mean_features_scores, -1).squeeze()
    mean_features_scores = np.sort(mean_features_scores, -1).squeeze()
    mean_mask = mean_features_scores > 0
    mean_features_indices = mean_sorted_indices[mean_mask]
    mean_features_scores = mean_features_scores[mean_mask]
    mean_features_idx_scores = list(zip(mean_features_indices, mean_features_scores))
    mean_top_10 = mean_features_idx_scores[-k:]
    feature_names = vectorizer.get_feature_names()
    mean_top_10 = [(feature_names[i], s) for i, s in mean_top_10]
    return list(reversed(mean_top_10))

def get_peculiars(l, ll):

    w1 = {w for w, _ in l}
    w2 = {w for w, _ in ll}
    lmll = {(w,s) for w, s in l if w not in w2}
    common = {w  for w, s in l if w in w2}
    return lmll, common

def get_peculiars_all(word_scores_list):
    peculiars = defaultdict(set)
    for i in range(len(word_scores_list)):
        l = word_scores_list[i]
        peculiars[i] = set(l)
        l_w = {w for w,_ in l}
        for j in range(0, len(word_scores_list)):
            if i == j:
                continue
            ll = word_scores_list[j]
            l_peculiars, common_words = get_peculiars(l, ll)
            peculiars[i].update(l_peculiars)
            peculiars[i] = {(w, s) for w, s in peculiars[i] if w not in common_words}
            #peculiars[i] = peculiars[i] - ll_peculiars if len(peculiars[i]) > 0 else l_peculiars
            #peculiars[j] = peculiars[j] - l_peculiars if len(peculiars[j]) > 0 else ll_peculiars
    return [peculiars[i] for i in range(len(peculiars))]

attribute_dicts = {'gender':["0", "1", "2"], 'respondent':['person',
'public_entity', 'organization', 'facility', 'other'], 'decisionDirection':['liberal','conservative'], 'age':['0', '1', '2', '3']}
dataset_versions = {'ecthr':'1.0', 'scotus':'0.4'}
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['ecthr', 'scotus'], required=True)
    parser.add_argument('--attribute', required=True)
    parser.add_argument('--log_dir_root', required = True)
    parser.add_argument('--dataset_root', required = False, default='data/datasets/')
    parser.add_argument('--weight_by_params', action='store_true', default=False)
    parser.add_argument('--params_path', default=None)
    parser.add_argument('--to_latex', default=False, action = 'store_true')
    
    args = parser.parse_args()
    
    to_latex = args.to_latex
    dataset_name = args.dataset #'ecthr'
    log_root = args.log_dir_root
    version = dataset_versions[dataset_name] #"1.0"
    attribute = args.attribute #'age'
    attribute_vals = attribute_dicts[attribute]
    weight_by_params = args.weight_by_params
    model_params_path = args.params_path #'logs_final_tfidf_regressor/ecthr/ERM/age/seed_1/ecthr_seed:1_epoch:best_model.pth'
    print(f'weight_by_params={weight_by_params}')
    print(f'model_params_path={model_params_path}')

    params=None
    if weight_by_params:
        assert model_params_path is not None or print(f"weight_by_params is True but no path is specified for the parameters (--params_path)")
        params = torch.load(model_params_path, map_location='cpu')['algorithm']['model.classifier.weight']

    logdir = os.path.join(log_root, dataset_name, 'ERM', attribute, 'seed_1')
    outdir = os.path.join(log_root, dataset_name, 'ERM', attribute, 'seed_1', 'interpretability')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = os.path.join(outdir, f'top_features_weight_by_params={weight_by_params}.txt')

    vectorizer_path = os.path.join(args.dataset_root, f'{dataset_name}_v{version}', 'tfidf_tokenizer_3grams_10000.pkl')
    tops = list()
    dataset = get_dataset(dataset_name, group_by_fields=[attribute,])
    val2idx = {x:int(x) for x in attribute_vals} if dataset_name == 'ecthr' else dataset.attribute2value2idx[attribute]

    for att_val in attribute_vals:
        aux = average_features(dataset, logdir, vectorizer_path, attribute,
                att_val, val2idx, weight_by_params, params,  k = 500)
        tops.append(aux)


    peculiars = get_peculiars_all(tops)
    d = defaultdict(list)
    # #print([sorted(x, key=lambda x : -x[-1]) for x in peculiars])

    for i, att_val in enumerate(attribute_vals):
        #words, scores = zip(*peculiars[i])
        #scores = np.array(scores)
        #scores = (np.e ** scores) / np.sum(np.e ** scores)
        #peculiars[i] = set(zip(words, scores))
        i_tops = sorted(peculiars[i], key=lambda x : -x[1])[:20]
        d[att_val].extend([w for w, _ in i_tops])
        x = [s for _, s in i_tops]
        d[f'{att_val}_score'].extend(x)#[s for _, s in sorted(i_tops, key=lambda x: -x[1])[:10]])


    df = pd.DataFrame.from_dict(d)
    out = tabulate(df, headers='keys', tablefmt='psql')
    print(out)
    with open(outfile, 'w') as writer:
        writer.write(out)

    print(f'results print on {outfile}')
    if to_latex:
        latex_outfile = outfile.replace('.txt', '.tex')
        with open(latex_outfile, 'w') as writer:
            writer.write(df.round(3).to_latex(index=False))
        print(f'tex table written at {latex_outfile}')

