from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


path = "/home/npf290/dev/fairlex-wilds/data/case.law/case.law.all.txt"

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=path, vocab_size=30_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model(".", "caselaw_bert")