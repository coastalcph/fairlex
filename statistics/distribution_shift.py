import pandas as pd
from dataloaders import get_dataset
from argparse import ArgumentParser
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon

def _get_label_distribution(dataset_df, y_size, dataset_name):
    if len(dataset_df) == 0:
        return None
    label_column_name = 'labels' if 'labels' in dataset_df.columns else 'label'
    labels = dataset_df[label_column_name].tolist()
    if isinstance(labels[0], list):
        if dataset_name == 'ecthr':
            labels = [([1] if sum(l) == 0 else [0]) + l for l in labels]
        dataset_labels = np.argmax(np.stack(labels, 0), -1)
    else:
        dataset_labels = np.array(labels)
    dataset_label_freq = Counter(dataset_labels)
    dataset_label_distr = []
    for i in range(y_size):
        dataset_label_distr.append(dataset_label_freq[i])
    den = sum(dataset_label_distr)
    return np.array(dataset_label_distr) / den

def get_label_distribution(dataset_df, y_size, dataset_name, p_attribute=None, attribute_values = None):
    distributions_by_av = list()
    if p_attribute is not None:
        for av in attribute_values:
            subset = dataset_df[dataset_df[p_attribute]==av]
            subset_label_distr = _get_label_distribution(subset, y_size, dataset_name)
            distributions_by_av.append(subset_label_distr)
    else:
        distributions_by_av.append(_get_label_distribution(dataset_df, y_size, dataset_name))

    return distributions_by_av

def compute_distribution_divergence(dataset_name, divergence_fun, p_attribute = None):
    dataset = get_dataset(dataset_name, group_by_fields=[p_attribute])
    attribute_values = None
    if p_attribute is not None:
        attribute_values = set(dataset.data_df[p_attribute].tolist())

    train_set = dataset.data_df[dataset.data_df['data_type'] == 0]
    dev_set = dataset.data_df[dataset.data_df['data_type'] == 1]
    test_set = dataset.data_df[dataset.data_df['data_type'] == 2]

    training_label_distrs = get_label_distribution(train_set, dataset._y_size, dataset_name, p_attribute, attribute_values)
    dev_label_distrs = get_label_distribution(dev_set, dataset._y_size, dataset_name, p_attribute, attribute_values)
    test_label_distrs = get_label_distribution(test_set, dataset._y_size, dataset_name, p_attribute, attribute_values)

    p, q, tt_att_val  = zip(*[(a, b, av) for a, b, av in zip(training_label_distrs, test_label_distrs, attribute_values) if a is not None and b is not None])
    train_test_div = divergence_fun(p, q)
    p, q, td_att_val = zip(*[(a, b, av) for a, b, av in zip(training_label_distrs, dev_label_distrs, attribute_values) if a is not None and b is not None])
    train_dev_div = divergence_fun(p, q)
    p, q, dt_att_val = zip(*[(a, b, av) for a, b, av in zip(dev_label_distrs, test_label_distrs, attribute_values) if a is not None and b is not None])
    dev_test_div = divergence_fun(p, q)
    return {'train-test divergence':list(zip(train_test_div, tt_att_val)), 'train-dev divergence': list(zip(train_dev_div, td_att_val)), 'dev-test divergence': list(zip(dev_test_div, dt_att_val))}

def jsd(d1, d2):
    return jensenshannon(d1, d2, axis=-1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--divergence_measure', choices=['jsd'], default = 'jsd') #jsd =jensen shannon divergence
    parser.add_argument('--protected_attribute', required=False, default=None)

    args = parser.parse_args()
    dataset_name = args.dataset
    divergence_fun_name = args.divergence_measure
    p_attribute = args.protected_attribute
    divergence_fun = None
    if divergence_fun_name == 'jsd':
        divergence_fun = jsd
    else:
        raise RuntimeError(f'Unrecognised divergence measure {divergence_fun_name}')
    distribution_divergence = compute_distribution_divergence(args.dataset, divergence_fun, p_attribute)
    for k, vals in distribution_divergence.items():
        print(f'{k}={" ".join([str(i) + ":" + str(s) for s,i in vals])}')
