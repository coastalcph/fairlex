import matplotlib.pyplot as plt
from dataloaders import get_dataset
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained('/home/iliasc/fairlex-wilds/data/models/mini-longformer')

for dataset in ['ecthr', 'scotus', 'fscs', 'pc']:
    dataset = get_dataset(dataset)
    text_lengths = []
    for document in dataset:
        text_lengths.append(len(TOKENIZER.tokenize(document['text'])))
    plt.hist(text_lengths, bins=50, range=(0, 6000), alpha=0.5, edgecolor='k', label=f'{dataset.upper()}')

plt.legend(loc='upper right')
plt.xlabel('Number of sub-word units (tokens)')
plt.ylabel('Number of pairs')
plt.show()
