from configs.supported import supported_datasets

from wilds import get_dataset as wilds_get_dataset
def get_dataset(dataset, version=None, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        version (str): Dataset version number, e.g., '1.0'.
                       Defaults to the latest version.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {supported_datasets}.')

    if dataset == 'ecthr':
        from dataloaders.ecthr_dataset import ECtHRDataset
        return ECtHRDataset(version=version, **dataset_kwargs)
    elif dataset == 'ledgar':
        from dataloaders.ledgar_dataset import LEDGARDataset
        return LEDGARDataset(version=version, **dataset_kwargs)
    elif 'scotus' in dataset:
        from dataloaders.scotus_dataset import ScotusDataset
        return ScotusDataset(version=version, **dataset_kwargs)
    elif dataset == 'fscs':
        from dataloaders.fscs_dataset import FSCSDataset
        return FSCSDataset(version=version, **dataset_kwargs)
    else: 
        return wilds_get_dataset(dataset, version, **dataset_kwargs)
    
