from diff_mask.model import MetaModel
import os
import torch

from data import MODELS_DIR

DATASET = 'ecthr'
ATTRIBUTE = 'age'
model_path = os.path.join(MODELS_DIR, 'ecthr-mini-longformer')
checkpoint_path = f'/home/iliasc/fairlex-wilds/final_logs/{DATASET}/ERM/{ATTRIBUTE}/seed_1/{DATASET}_seed:1_epoch:best_model.pth'
model = MetaModel(model_path=model_path)
state = torch.load(checkpoint_path)
model.load_state_dict(state['algorithm'])