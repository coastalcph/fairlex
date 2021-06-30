from transformers import LongformerForMaskedLM, AutoModelForMaskedLM, AutoTokenizer
import copy
import torch
import argparse

def parse_int(v):
    return int(v)

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model', default='nlpaueb/legal-bert-small-uncased')
    parser.add_argument('--transformation', default='longformer', choices=['longformer', 'longbert'])
    # Model arguments
    parser.add_argument('--max_pos', default=4096, type=parse_int)
    parser.add_argument('--attention_window', default=128, type=parse_int)

    run_config = parser.parse_args()

    bert = AutoModelForMaskedLM.from_pretrained(run_config.model)
    tokenizer = AutoTokenizer.from_pretrained(run_config.model, model_max_length=run_config.max_pos)

    # extend position embedding
    config = bert.config
    tokenizer.model_max_length = run_config.max_pos
    tokenizer.init_kwargs['model_max_length'] = run_config.max_pos
    current_max_pos, embed_size = bert.bert.embeddings.position_embeddings.weight.shape
    run_config.max_pos += 2
    config.max_position_embeddings = run_config.max_pos
    assert run_config.max_pos > current_max_pos

    new_pos_embed = bert.bert.embeddings.position_embeddings.weight.new_empty(run_config.max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < run_config.max_pos - 1:
        if k + step >= run_config.max_pos:
            new_pos_embed[k:] = bert.bert.embeddings.position_embeddings.weight[2:(run_config.max_pos + 2 - k)]
        else:
            new_pos_embed[k:(k + step)] = bert.bert.embeddings.position_embeddings.weight[2:]
        k += step
    bert.bert.embeddings.position_embeddings.weight.data = new_pos_embed
    bert.bert.embeddings.position_ids.data = torch.tensor([i for i in range(run_config.max_pos)]).reshape(1, run_config.max_pos)

    if run_config.transformation == 'longformer':
        # add global attention
        config.attention_window = [run_config.attention_window] * config.num_hidden_layers
        for i in range(len(bert.bert.encoder.layer)):
            bert.bert.encoder.layer[i].attention.self.query_global = copy.deepcopy(bert.bert.encoder.layer[i].attention.self.query)
            bert.bert.encoder.layer[i].attention.self.key_global = copy.deepcopy(bert.bert.encoder.layer[i].attention.self.key)
            bert.bert.encoder.layer[i].attention.self.value_global = copy.deepcopy(bert.bert.encoder.layer[i].attention.self.value)

        lfm = LongformerForMaskedLM(config)
        lfm.longformer.load_state_dict(bert.bert.state_dict())
        lfm.lm_head.dense.load_state_dict(bert.cls.predictions.transform.dense.state_dict())
        lfm.lm_head.layer_norm.load_state_dict(bert.cls.predictions.transform.LayerNorm.state_dict())
        lfm.lm_head.decoder.load_state_dict(bert.cls.predictions.decoder.state_dict())
        lfm.lm_head.bias = copy.deepcopy(bert.cls.predictions.bias)

        # Extra config parameters
        config.attention_mode = "longformer"
        config.model_type = "longformer"

        lfm.save_pretrained('legal-longformer')
        tokenizer.save_pretrained('legal-longformer')

    else:
        bert.save_pretrained('legal-longbert')
        tokenizer.save_pretrained('legal-longbert')


if __name__ == '__main__':
    main()
