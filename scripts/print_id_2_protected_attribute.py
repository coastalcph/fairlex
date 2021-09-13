from dataloaders.scotus_dataset import ScotusDataset


def print_id2protected_attribute():
    dataset = ScotusDataset("official", "decisionDirection")
    pa_id2str = dataset.protected_attribute_id2str
    for k, v in sorted(pa_id2str.items(), key=lambda x: x[0]):
        print(f'{k}\t{v}')

if __name__ == '__main__':
    print_id2protected_attribute()
    