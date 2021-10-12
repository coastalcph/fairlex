dataset_defaults = {
    'ecthr': {
        'split_scheme': 'official',
        'model': 'ecthr-mini-roberta',
        'train_transform': 'hier-bert',
        'eval_transform': 'hier-bert',
        'max_token_length': 4096,
        'max_segments': 32,
        'max_segment_length': 128,
        'loss_function': 'binary_cross_entropy',
        'algo_log_metric': 'multi-label-f1',
        'val_metric': 'F1-macro_all',
        'batch_size': 16,
        'lr': 3e-5,
        'weight_decay': 0,
        'n_epochs': 100,
        'n_groups_per_batch': 2,
        'groupby_fields': ['defendant'],
        'irm_lambda': 0.5,
        'coral_penalty_weight': 0.5,
        'adv_lambda': 0.5,
        'rex_beta': 0.5,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
    'fscs': {
        'split_scheme': 'official',
        'model': 'fscs-mini-xlm-roberta',
        'train_transform': 'hier-bert',
        'eval_transform': 'hier-bert',
        'max_token_length': 2048,
        'max_segments': 32,
        'max_segment_length': 64,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'multi-class-f1',
        'val_metric': 'F1-macro_all',
        'batch_size': 32,
        'lr': 3e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['defendant'],
        'irm_lambda': 0.5,
        'coral_penalty_weight': 0.5,
        'adv_lambda': 0.5,
        'rex_beta': 0.5,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
    'spc': {
        'split_scheme': 'official',
        'model': 'spc-mini-xlm-roberta-v1',
        'train_transform': 'hier-bert',
        'eval_transform': 'hier-bert',
        'max_token_length': 2048,
        'max_segments': 32,
        'max_segment_length': 128,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'multi-class-f1',
        'val_metric': 'F1-macro_all',
        'batch_size': 32,
        'lr': 3e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['gender'],
        'irm_lambda': 0.5,
        'coral_penalty_weight': 0.5,
        'adv_lambda': 0.5,
        'rex_beta': 0.5,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
    'scotus': {
        'split_scheme': 'official',
        'model': 'scotus-mini-roberta',
        # 'train_transform': 'hier-bert',#'longformer',
        # 'eval_transform': 'hier-bert', #'longformer',
        'max_token_length': 4096,
        'max_segments': 32,
        'max_segment_length': 128,
        'loss_function': 'binary_cross_entropy',
        'batch_size': 12,
        # 'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        },
        'algo_log_metric': 'multi-class-f1-for-binary-loss',
        'val_metric': 'F1-macro_all',
        'val_metric_decreasing': False,
    }
}

