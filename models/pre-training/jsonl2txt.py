from data import DATA_DIR
import glob
import json
import os

filenames = glob.glob(os.path.join(DATA_DIR, 'datasets', 'ecthr_v1.0', '*.jsonl'))

with open(os.path.join(DATA_DIR, 'datasets', 'ecthr_v1.0', 'ecthr.text.raw'), 'w') as out_file:
    for filename in filenames:
        with open(filename) as in_file:
            for line in in_file.readlines():
                data = json.loads(line)
                out_file.write('\n'.join(data['facts']) + '\n')
