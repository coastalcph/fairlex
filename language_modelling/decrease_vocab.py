import copy
import json
import os
import re
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from torch.nn import Parameter
import string
SAVE_DIR = os.path.join('my-xlm-roberta-base')

# CLEAN FOLDER
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)

# SAVE ORIGINAL XLM-ROBERTA TOKENIZER IN THE NEW FORMAT
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
config = AutoConfig.from_pretrained('xlm-roberta-base')
tokenizer.save_pretrained(SAVE_DIR, legacy_format=False)
config.save_pretrained(SAVE_DIR)

# FIND USABLE TOKENS IN VOCABULARY
# Keep only numbers, latin script and other chars than make sense for DE, FR, IT.
VOCAB = re.compile(f'[a-zA-Z0-9ÊõüèëØÅçéá̊òąàôœå≅êûßÖíä§ÂÇñǗÈË"ã€øαÉÀĄïṣùâóÎÄúÓæöμÔìî]+')
eu_tokens = []
usable_tokens = []
not_usable = []
usable_ids = []
for original_token, id in tokenizer.vocab.items():
    token = original_token.translate(str.maketrans('', '', string.punctuation + '▁–⁄°€')).lower()
    if VOCAB.fullmatch(token) or len(token) == 0:
        usable_tokens.append(original_token)
        usable_ids.append(id)
    else:
        not_usable.append(original_token)

print(f'USABLE VOCABULARY: {len(usable_ids)}/{len(tokenizer.vocab)} ({(len(usable_ids)*100)/len(tokenizer.vocab):.1f}%)')


# UPDATE TOKENIZER VOCABULARY
usable_tokens = set(usable_tokens)
with open(os.path.join(SAVE_DIR, 'tokenizer.json')) as file:
    tokenizer_data = json.load(file)
    tokenizer_data['model']['vocab'] = [token for token in tokenizer_data['model']['vocab'] if token[0] in usable_tokens]
    tokenizer_data['added_tokens'][-1]['id'] = len(tokenizer_data['model']['vocab'])

# SAVE TOKENIZER JSON
with open(os.path.join(SAVE_DIR, 'tokenizer.json'), 'w') as file:
    json.dump(tokenizer_data, file)

# HACK XLM-ROBERTA
print('HACK XLM-ROBERTA')
# LOAD XLM-ROBERTA
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
eu_model_pt = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
# COLLECT USABLE (EMBEDDINGS + LM HEAD) WEIGHTS
usable_ids = set(usable_ids)
embeddings = copy.deepcopy([embed for idx, embed in enumerate(eu_model_pt.roberta.embeddings.word_embeddings.weight.detach().numpy()) if idx in usable_ids])
lm_head_bias = copy.deepcopy([embed for idx, embed in enumerate(eu_model_pt.lm_head.bias.detach().numpy()) if idx in usable_ids])
lm_head_decoder_bias = copy.deepcopy([embed for idx, embed in enumerate(eu_model_pt.lm_head.decoder.bias.detach().numpy()) if idx in usable_ids])
lm_head_decoder_weight = copy.deepcopy([embed for idx, embed in enumerate(eu_model_pt.lm_head.decoder.weight.detach().numpy()) if idx in usable_ids])
# REASSIGN USABLE WEIGHTS TO (EMBEDDINGS + LM HEAD) LAYERS
eu_model_pt.resize_token_embeddings(len(usable_ids))
eu_model_pt.roberta.embeddings.word_embeddings.weight = Parameter(torch.as_tensor(embeddings))
eu_model_pt.lm_head.bias = Parameter(torch.as_tensor(lm_head_bias))
eu_model_pt.lm_head.decoder.weight = Parameter(torch.as_tensor(lm_head_decoder_weight))
eu_model_pt.lm_head.decoder.bias = Parameter(torch.as_tensor(lm_head_decoder_bias))

# SAVE MODEL AND TOKENIZER
eu_model_pt.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR, legacy_format=False)


# TEST MODEL AS LANGUAGE MODEL
print('INFERENCE')
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
final_model = AutoModelForMaskedLM.from_pretrained(SAVE_DIR)

with torch.no_grad():
    print(tokenizer.decode(torch.argmax(
        final_model(input_ids=tokenizer('Her <mask> is beautiful.',
                                        return_tensors='pt').input_ids).logits, dim=-1).numpy()[0]))
    print(tokenizer.decode(torch.argmax(
        final_model(input_ids=tokenizer('A <mask> sunny holiday.',
                                        return_tensors='pt').input_ids).logits, dim=-1).numpy()[0]))
    print(tokenizer.decode(torch.argmax(
        final_model(input_ids=tokenizer('He played <mask> guitar, while the other guy was playing piano.',
                                        return_tensors='pt').input_ids).logits, dim=-1).numpy()[0]))
    print(tokenizer.decode(torch.argmax(
        final_model(input_ids=tokenizer('Paris is the <mask> of France.',
                                        return_tensors='pt').input_ids).logits, dim=-1).numpy()[0]))
    print(tokenizer.decode(torch.argmax(
        final_model(input_ids=tokenizer('Ο πρόεδρος του <mask>.',
                                        return_tensors='pt').input_ids).logits, dim=-1).numpy()[0]))
