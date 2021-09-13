import json
import os
import shutil

from transformers import LongformerForMaskedLM, AutoModelForMaskedLM
import copy
import torch
import argparse


def parse_int(v):
    return int(v)


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model', default='eu-mini-xlm-roberta')
    parser.add_argument('--transformation', default='longformer', choices=['longformer', 'longbert'])
    parser.add_argument('--output_model_name', default='mini-xlm-longformer')
    # Model arguments
    parser.add_argument('--max_pos', default=2048, type=parse_int)
    parser.add_argument('--attention_window', default=64, type=parse_int)

    args = parser.parse_args()

    # load model
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    # extend position embedding
    config = model.config
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    args.max_pos += 2
    config.max_position_embeddings = args.max_pos
    assert args.max_pos > current_max_pos

    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(args.max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < args.max_pos - 1:
        if k + step >= args.max_pos:
            new_pos_embed[k:] = model.roberta.embeddings.position_embeddings.weight[2:(args.max_pos + 2 - k)]
        else:
            new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(args.max_pos)]).reshape(1, args.max_pos)

    if args.transformation == 'longformer':
        # add global attention
        config.attention_window = [args.attention_window] * config.num_hidden_layers
        for i in range(len(model.roberta.encoder.layer)):
            model.roberta.encoder.layer[i].attention.self.query_global = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.query)
            model.roberta.encoder.layer[i].attention.self.key_global = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.key)
            model.roberta.encoder.layer[i].attention.self.value_global = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.value)

        lfm = LongformerForMaskedLM(config)
        lfm.longformer.load_state_dict(model.roberta.state_dict())
        lfm.lm_head.dense.load_state_dict(model.lm_head.dense.state_dict())
        lfm.lm_head.layer_norm.load_state_dict(model.lm_head.layer_norm.state_dict())
        lfm.lm_head.decoder.load_state_dict(model.lm_head.decoder.state_dict())
        lfm.lm_head.bias = copy.deepcopy(model.lm_head.bias)

        # extra config parameters
        config.attention_mode = "longformer"
        config.model_type = "longformer"

        # save model
        lfm.save_pretrained(args.output_model_name)

        # save tokenizer files
        for filename in ['tokenizer.json', 'merges.txt', 'vocab.json', 'sentencepiece.bpe.model',
                         'tokenizer_config.json', 'special_tokens_map.json']:
            if os.path.exists(f'{args.model}/{filename}'):
                shutil.copy(f'{args.model}/{filename}', f'{args.output_model_name}/{filename}')

        # amend configuration file
        with open(f'{args.output_model_name}/config.json') as config_file:
            configuration = json.load(config_file)
            configuration['model_type'] = "longformer"
        with open(f'{args.output_model_name}/config.json', 'w') as config_file:
            json.dump(configuration, config_file)

    else:
        # save model and tokenizer
        model.save_pretrained(args.output_model_name)
        for filename in ['tokenizer.json', 'merges.txt', 'vocab.json', 'tokenizer_config.json', 'special_tokens_map.json']:
            shutil.copy(f'{args.model}/{filename}', f'{args.output_model_name}/{filename}')


if __name__ == '__main__':
    main()
