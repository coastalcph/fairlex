from dataclasses import dataclass
import datasets
from datasets.info import DatasetInfo
from datasets.load import load_dataset
from datasets.utils.download_manager import DownloadManager
import os
import jsonlines
import json

_DESCRIPTION = """Dataset extracted from case laws of Supreme Court of United States."""

_VERSION = "0.0.1"
_DOWNLOAD_URL='https://sid.erda.dk/share_redirect/g9FbjeorfO'

@dataclass
class SCOTUSConfig(datasets.BuilderConfig):
    version:str=None
    name:str = None
    extended_name: str = None


class SCOTUS(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SCOTUSConfig(
            name='SCOTUS',
            extended_name = 'Supreme Court of United States Dataset',
            version=datasets.Version(_VERSION, ""),
        )
    ]

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "json_attributes": datasets.Value("string"),
                    "labels": datasets.Value("string")
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        downloaded_file = dl_manager.download_and_extract(_DOWNLOAD_URL)
        dataset_root_folder = os.path.join(downloaded_file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root_folder,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root_folder,
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root_folder,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, dataset_root, split, **kwargs):
        path = os.path.join(dataset_root, 'scotus', f'scotus.{split}.jsonl')
        with jsonlines.open(path) as lines:
            for line in lines:
                attributes = line['attributes']
                attributes_json = json.dumps(attributes)
                iid = attributes['docketId']
                text = line['text']
                labels = str(line['labels'])
                yield iid, {'text': text, 'id': iid, 'labels':labels, 'json_attributes':attributes_json}
