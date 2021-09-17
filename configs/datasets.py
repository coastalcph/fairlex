dataset_defaults = {
    'ecthr': {
        'split_scheme': 'official',
        'model': 'ecthr-mini-longformer',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'max_token_length': 4096,
        'loss_function': 'binary_cross_entropy',
        'algo_log_metric': 'multi-label-f1',
        'val_metric': 'F1-macro_all',
        'batch_size': 16,
        'lr': 3e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['defendant'],
        'irm_lambda': 1.0,
        'coral_penalty_weight': 1.0,
        'adv_lambda': 0.1,
        'rex_beta': 1.0,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
    'fscs': {
        'split_scheme': 'official',
        'model': 'fscs-mini-xlm-longformer',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'max_token_length': 2048,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'multi-class-f1',
        'val_metric': 'F1-macro_all',
        'batch_size': 32,
        'lr': 3e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['defendant'],
        'irm_lambda': 1.0,
        'coral_penalty_weight': 1.0,
        'adv_lambda': 0.1,
        'rex_beta': 1.0,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
}

