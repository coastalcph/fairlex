from typing import Union
import pandas as pd
from scipy.sparse.csr import csr_matrix
import numpy as np
def eval_by_group(model, dataset:pd.DataFrame, protected_attribute_name:str, text_transformer, val_mapping:dict=None):
        f1_by_group = dict()
        for val in dataset[protected_attribute_name].unique():
            group_dataset = dataset[dataset[protected_attribute_name]==val]
            if val_mapping is not None:
                val = val_mapping.get(val, val)
            f1_by_group[val] = eval(model, group_dataset, text_transformer)
        return f1_by_group

def eval(model, dataset:Union[pd.DataFrame, csr_matrix], text_transformer, labels = None):
    from sklearn import metrics
    assert type(dataset) != csr_matrix or labels is not None
    if type(dataset) == pd.DataFrame:
        labels = dataset['labels']
        dataset = text_transformer.transform(dataset['text'])
    predictions = model.predict(dataset)
    return metrics.f1_score(predictions,np.stack(labels), average='macro')