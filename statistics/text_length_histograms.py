import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import tqdm

for dataset, color, model in zip(['ecthr', 'scotus', 'fscs'], ['blue', 'red', 'orange'], ['mini-longformer', 'mini-longformer', 'mini-xlm-longformer']):
    text_lengths = []
    TOKENIZER = AutoTokenizer.from_pretrained(f'/home/iliasc/fairlex-wilds/data/models/{model}')
    with open(f'/home/iliasc/fairlex-wilds/data/datasets/{dataset}_v1.0/{dataset}_dump.txt') as file:
        for line in tqdm.tqdm(file.readlines()):
            text_lengths.append(len(TOKENIZER.tokenize(line)))
    plt.hist(text_lengths, bins=50, range=(0, 6000), alpha=0.5, edgecolor='k', color=color)
    plt.xlabel('Number of sub-word units (tokens)')
    plt.ylabel('Number of documents')
    plt.title(dataset.upper())
    plt.show()
