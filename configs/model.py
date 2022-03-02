model_defaults = {
    'coastalcph/fairlex-ecthr-minilm': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'coastalcph/fairlex-scotus-minilm': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'coastalcph/fairlex-fscs-minilm': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'coastalcph/fairlex-cail-minilm': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'logistic_regression': {
        'optimizer': 'Adam',
        'max_grad_norm': 1.0,
    },
}
