from argparse import ArgumentParser
from dataloaders.scotus_dataset import ScotusDataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--model_answers_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed_no", default=1, type=int)
    parser.add_argument('--attribute_name', default=None)
    parser.add_argument('--attribute_value', default=None)
    args = parser.parse_args()
    attribute_filter = (args.attribute_name, args.attribute_value)
    if any(x is None for x in attribute_filter):
        attribute_filter = None
    ScotusDataset.dump_error_dataset(
        args.dataset_root, args.model_answers_root,  args.seed_no, args.out_dir,attribute_filter=attribute_filter
    )
