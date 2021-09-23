model_defaults = {
    'mini-longformer': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'mini-roberta': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'ecthr-mini-longformer-v2': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'scotus-mini-longformer': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'mini-xlm-longformer': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'fscs-mini-xlm-longformer': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'regressor':
    {
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'in_dim': 10_000,
        # 'tfidf_vectorizer_path': '/home/npf290/dev/fairlex-wilds/data/scotus_v0.4/tfidf_tokenizer.pkl'
    }
}
