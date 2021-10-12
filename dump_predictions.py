import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import jsonlines
from configs.utils import populate_defaults
from dataloaders import get_dataset
from transforms import initialize_transform
from configs.datasets import dataset_defaults
from argparse import ArgumentParser
from dataloaders.scotus_dataset import ScotusDataset
from models.initializer import initialize_hierbert_model
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from dataloaders.scotus_attribute_mapping import ISSUE_AREA_MAPPING

DATASET_VERSION = {'ecthr':'1.0', 'scotus':'0.4'}
MODELS_DIR='data/models'


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




def dump_predictions(weights_path:str, dataset_name:str, out_path:str):
    config = dataset_defaults[dataset_name]
    config['train_transform']='hier-bert'
    config = dotdict(config)
    config.model = os.path.join(MODELS_DIR, config.model)
    dataset = get_dataset(dataset_name, version = DATASET_VERSION[dataset_name])
    model = initialize_hierbert_model(config, dataset._y_size)
    print(config)
    print(weights_path)
    weights = torch.load(weights_path, map_location='cpu')['algorithm']
    weights = {k.replace('model.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights)
    transform = initialize_transform(transform_name = config.train_transform, config=config)
    training_set = dataset.get_subset('train', transform=transform)
    loader = DataLoader(training_set,
               shuffle=False, # Shuffle training dataset
               sampler=None,
               collate_fn=dataset.collate,
               batch_size=config.batch_size)
    count = 0
    with open(out_path, 'w') as writer:
        for batch in tqdm(loader):
            x, y_true, metadata = batch
            x = x.to('cpu')
            y_true = torch.argmax(y_true.to('cpu'), -1)
            output = model(x)
            y_hat = torch.argmax(output, -1)
            y_hat = y_hat.tolist()
            writer.write('\n'.join(map(str, y_hat)) + '\n')
    print(count)

def create_new_dataset(predictions_path, dataset_json_path, out_path):
    data = ScotusDataset.load_raw_dataset(dataset_json_path)
    lid2l = {str(v):str(k) for k, v in ISSUE_AREA_MAPPING.items()}
    print(lid2l)
    predictions = list()
    with open(predictions_path) as lines:
        for line in lines:
            line = line.strip()
            l = lid2l[line]
            predictions.append((l, line))
    assert len(data) == len(predictions) or print(len(data), len(predictions))
    with jsonlines.open(out_path, 'w') as writer :
        for ex, (label, label_id) in zip(data, predictions):
            ex['label'] = label
            ex['label_id'] = label_id
            writer.write(ex)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights_path', required=True)
    parser.add_argument('--dataset_name', choices=['scotus', 'ecthr'], required=True)
    parser.add_argument('--out_path', required=True)
    args = parser.parse_args()
    dump_predictions(args.weights_path, args.dataset_name, args.out_path)
    dataset_version = '0.4' if args.dataset_name == 'scotus' else '1.0'
    create_new_dataset(args.out_path, f'data/datasets/{args.dataset_name}_v{dataset_version}/{args.dataset_name}.train.jsonl', 
                      '{args.dataset_name}_v{dataset_version}.hier-bert-answers.jsonl')
