from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from argparse import ArgumentParser

from dataloaders import get_dataset
import pickle as pkl
import logging
logger = logging.getLogger('vectorizers')
logger.setLevel(logging.INFO)

def train_tfidf_vectorizer(
    dataset, 
    outpath,
    ngram_max=3, 
    lowercase=True, 
    max_features = 10_000, 
    stop_words='english'
    ):
    
    vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, ngram_max),
            lowercase=lowercase,
            max_features=max_features)
    __fit_and_save_vectorizer(vectorizer, outpath, dataset)
    
def train_bow_vectorizer(
    dataset_name, 
    outpath,
    ngram_range=(1, 3), 
    lowercase=True, 
    num_features = 10_000, 
    stop_words='english'
    ):
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range,
    lowercase=lowercase, max_features=num_features)
    __fit_and_save_vectorizer(vectorizer, outpath, dataset_name)

def __fit_and_save_vectorizer(vectorizer, outpath, dataset_name):
    dataset = get_dataset(dataset_name)
    logger.info('Fitting TF-IDF vectorizer')
    if 'text' in dataset.get_subset('train'):
        vectorizer.fit(dataset.get_subset('train')["text"])
    else:
        vectorizer.fit([x[0] for x in dataset.get_subset('train')])
    logger.info(f'Saving vectorizer to {outpath}')
    with open(outpath, 'wb') as writer:
        pkl.dump(vectorizer, writer)
    logger.info('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', choices=['tfidf', 'bow'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--ngram_max', type=int, default=3)
    parser.add_argument('--lowercase', action='store_true', default=False)
    parser.add_argument('--max_features', type=int, default=10_000)
    args = parser.parse_args()
    vargs = dict(vars(args))
    del vargs['type']
    if args.type == 'tfidf':
        train_tfidf_vectorizer(**vargs)
    elif args.type == 'bow':
        train_bow_vectorizer(**vargs)
    




        