import os
import jsonlines
import lzma
import json
from transformers.models.auto import AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool

def parse_file(path):
    bar = tqdm()
    with lzma.open(path) as reader, open(path + '.txt', 'w') as writer:
        for line in reader:
            data = json.loads(line)
            if 'puerto rico' in data['court']['name'].lower(): #documents in spanish
                continue
            for opinion in data['casebody']['data']['opinions']:
                text = opinion['text']
                writer.write(text + ' ')
            bar.update()
def parse(folder, tokenizer_name, outfile):
    all_paths = list()
    with open(outfile, 'w') as writer:
        for jurisdiction_folder in os.listdir(folder):
            if jurisdiction_folder.endswith('.zip'):
                continue
            if not os.path.isdir(os.path.join(folder, jurisdiction_folder)):
                continue

            jsonl_path = os.path.join(folder, jurisdiction_folder, 'data/data.jsonl.xz')
            
            all_paths.append(jsonl_path)
    
    # with Pool(16) as pool:
        
    #     res = pool.map_async(parse_file, all_paths)
    #     res.get()
    
    with open(outfile, 'w') as writer:
        for f in tqdm(all_paths, desc='merging'):
            f = f + '.txt'
            with open(f) as reader:
                for line in reader:
                    writer.write(line)


if __name__ == '__main__':
    parse('/home/npf290/dev/fairlex-wilds/data/case.law', None, '/home/npf290/dev/fairlex-wilds/data/case.law/case.law.all.txt')
