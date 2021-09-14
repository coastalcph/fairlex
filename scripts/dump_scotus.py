import json
import re
import os

filenames = ['scotus.train.jsonl', 'scotus.dev.jsonl']
rootdir = '/home/npf290/dev/fairlex-wilds/data/scotus_v0.3/'
with open('scotus_dump.txt', 'w') as out_file:
    for filename in filenames:
        filename = os.path.join(rootdir, filename)
        with open(filename) as file:
            for line in file.readlines():
                example = json.loads(line)
                out_file.write(' </s> '.join(re.split('\n{2,}', example['text'])).replace('\n', '') + '\n')