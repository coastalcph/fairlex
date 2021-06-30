import os
import zipfile
import requests
from data import DATA_DIR
_URL = "http://archive.org/download/ECtHR-NAACL2021/dataset.zip"
SAVE_DIR = os.path.join(DATA_DIR, 'ecthr_v1.0', 'dataset.zip')
DATASET_DIR = os.path.join(DATA_DIR, 'ecthr_v1.0')


r = requests.get(_URL, stream=True)
with open(SAVE_DIR, 'wb') as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)
z = zipfile.ZipFile(SAVE_DIR)
z.extractall(DATASET_DIR)
with open(os.path.join(DATASET_DIR, 'RELEASE_v1.0.txt'), 'w') as file:
    file.write('')




