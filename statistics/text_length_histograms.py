import json

import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import tqdm

for dataset, color, model in zip(['spc'], ['green'], ['mini-xlm-roberta']):
    text_lengths = []
    TOKENIZER = AutoTokenizer.from_pretrained(f'/home/iliasc/fairlex-wilds/data/models/{model}')
    with open(f'/home/iliasc/fairlex-wilds/data/datasets/{dataset}_v1.0/spc.jsonl') as file:
        for line in tqdm.tqdm(file.readlines()):
            text_lengths.append(len(TOKENIZER.tokenize(' '.join(json.loads(line)['text']))))
    plt.hist(text_lengths, bins=50, range=(0, 6000), alpha=0.5, edgecolor='k', color=color)
    plt.xlabel('Number of sub-word units (tokens)')
    plt.ylabel('Number of documents')
    plt.title(dataset.upper())
    plt.show()
