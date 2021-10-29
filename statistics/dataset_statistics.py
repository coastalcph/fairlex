from argparse import ArgumentParser
from collections import Counter
from dataloaders import ecthr_dataset as ecthr
from dataloaders import get_dataset


def get_attribute_frequencies_by_values_in_training(dataset, attribute):
    data_df = dataset.data_df[dataset.data_df['data_type'] == 0]
    values = sorted(list(set(data_df[attribute].tolist())))
    freqs = []
    for v in values:
        v_freq = len(data_df[data_df[attribute] == v])
        freqs.append(v_freq)
    sorted_items = sorted(zip(values, freqs), key = lambda x : x[0])
    return sorted_items


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--protected_attribute', required = True)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    protected_attribute = args.protected_attribute
    dataset = get_dataset(dataset_name, group_by_fields = [protected_attribute])
    freqs = get_attribute_frequencies_by_values_in_training(dataset, protected_attribute)
    if dataset_name == 'scotus':
        pa_indices = dataset.data_df[protected_attribute].tolist()
        pa_str = dataset.data_df[protected_attribute+'_str'].tolist()
        print(sorted(set(zip(pa_indices, pa_str)), key= lambda x : x[0]))
    elif dataset_name == 'ecthr':
        if protected_attribute == 'age':
            print([(v,k) for k, v in ecthr.AGE_GROUPS.items()])
        elif protected_attribute == 'gender':
            print([(v,k) for k, v in ecthr.GENDERS.items()])
        else:
            print([(0,'EAST'), (1, 'WEST')])
    elif dataset_name == 'fscs':
       pa_indices = dataset.data_df[protected_attribute]
       pa_str = dataset.data_df[protected_attribute.replace('_', ' ')]
       print(sorted(set(zip(pa_indices, pa_str)), key=lambda x:x[0]))
    print(freqs)
