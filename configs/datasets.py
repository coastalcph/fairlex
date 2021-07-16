dataset_defaults = {
    'ecthr': {
        'split_scheme': 'official',
        'model': 'legal-longformer',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'max_token_length': 4096,
        'loss_function': 'binary_cross_entropy',
        'algo_log_metric': 'multi-label-f1',
        'val_metric': 'F1-micro_all',
        'batch_size': 12,
        'lr': 2e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['defendant'],
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'adv_lambda': 0.01,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
    'ledgar': {
        'split_scheme': 'official',
        'model': 'nlpaueb/legal-bert-small-uncased',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'multi-class-f1',
        'val_metric': 'F1-micro_all',
        'batch_size': 32,
        'lr': 2e-5,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['industry'],
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'adv_lambda': 0.1,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    },
    'scotus': {
        'split_scheme': 'official',
        'model': 'legal-longformer',#'allenai/longformer-base-4096',
        'train_transform': 'bert',#'longformer',
        'eval_transform': 'bert', #'longformer',
        'max_token_length': 4096,
        'loss_function': 'binary_cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 8,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 10,
        'n_groups_per_batch': 2,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        },
        'val_metric': 'F1-macro_all',
        'val_metric_decreasing': False
    },
    'amazon': {
        'split_scheme': 'official',
        'model': 'distilbert-base-uncased',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 8,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 3,
        'n_groups_per_batch': 2,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 1.0,
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'bdd100k': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'loss_function': 'multitask_bce',
        'val_metric': 'acc_all',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'n_epochs': 10,
        'algo_log_metric': 'multitask_binary_accuracy',
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'process_outputs_function': 'binary_logits_to_pred',
    },
    'camelyon17': {
        'split_scheme': 'official',
        'model': 'densenet121',
        'model_kwargs': {'pretrained': False},
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'target_resolution': (96, 96),
        'loss_function': 'cross_entropy',
        'groupby_fields': ['hospital'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.01,
        'n_epochs': 5,
        'n_groups_per_batch': 2,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'celebA': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'model_kwargs': {'pretrained': True},
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['male', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum': 0.9},
        'scheduler': None,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 0.0,
        'n_epochs': 200,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'civilcomments': {
        'split_scheme': 'official',
        'model': 'distilbert-base-uncased',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['black', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'batch_size': 16,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'n_epochs': 5,
        'algo_log_metric': 'accuracy',
        'max_token_length': 300,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 10.0,
        'loader_kwargs': {
            'num_workers': 1,
            'pin_memory': True,
        },
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'fmow': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'oracle_training_set': False,
            'seed': 111,
            'use_ood_val': True
        },
        'model': 'densenet121',
        'model_kwargs': {'pretrained': True},
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'loss_function': 'cross_entropy',
        'groupby_fields': ['year',],
        'val_metric': 'acc_worst_region',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
        'batch_size': 64,
        'lr': 0.0001,
        'weight_decay': 0.0,
        'n_epochs': 50,
        'n_groups_per_batch': 8,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'algo_log_metric': 'accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'iwildcam': {
        'loss_function': 'cross_entropy',
        'val_metric': 'F1-macro_all',
        'model_kwargs': {'pretrained': True},
        'train_transform': 'image_base',
        'eval_transform': 'image_base',
        'target_resolution': (448, 448),
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'model': 'resnet50',
        'lr': 3e-5,
        'weight_decay': 0.0,
        'batch_size': 16,
        'n_epochs': 12,
        'optimizer': 'Adam',
        'split_scheme': 'official',
        'scheduler': None,
        'groupby_fields': ['location',],
        'n_groups_per_batch': 2,
        'irm_lambda': 1.,
        'coral_penalty_weight': 10.,
        'no_group_logging': True,
        'process_outputs_function': 'multiclass_logits_to_pred'
    },
    'ogb-molpcba': {
        'split_scheme': 'official',
        'model': 'gin-virtual',
        'model_kwargs': {'dropout':0.5}, # include pretrained
        'loss_function': 'multitask_bce',
        'groupby_fields': ['scaffold',],
        'val_metric': 'ap',
        'val_metric_decreasing': False,
        'optimizer': 'Adam',
        'batch_size': 32,
        'lr': 1e-03,
        'weight_decay': 0.,
        'n_epochs': 100,
        'n_groups_per_batch': 4,
        'irm_lambda': 1.,
        'coral_penalty_weight': 0.1,
        'no_group_logging': True,
        'process_outputs_function': None,
        'algo_log_metric': 'multitask_binary_accuracy',
    },
    'py150': {
        'split_scheme': 'official',
        'model': 'code-gpt-py',
        'loss_function': 'lm_cross_entropy',
        'val_metric': 'acc',
        'val_metric_decreasing': False,
        'optimizer': 'AdamW',
        'optimizer_kwargs': {'eps':1e-8},
        'lr': 8e-5,
        'weight_decay': 0.,
        'n_epochs': 3,
        'batch_size': 6,
        'groupby_fields': ['repo',],
        'n_groups_per_batch': 2,
        'irm_lambda': 1.,
        'coral_penalty_weight': 1.,
        'no_group_logging': True,
        'algo_log_metric': 'multitask_accuracy',
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'poverty': {
        'split_scheme': 'official',
        'dataset_kwargs': {
            'no_nl': False,
            'fold': 'A',
            'oracle_training_set': False,
            'use_ood_val': True
        },
        'model': 'resnet18_ms',
        'model_kwargs': {'num_channels': 8},
        'train_transform': 'poverty_train',
        'eval_transform': None,
        'loss_function': 'mse',
        'groupby_fields': ['country',],
        'val_metric': 'r_wg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'mse',
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma':0.96},
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 0.0,
        'n_epochs': 200,
        'n_groups_per_batch': 8,
        'irm_lambda': 1.0,
        'coral_penalty_weight': 0.1,
        'process_outputs_function': None,
    },
    'waterbirds': {
        'split_scheme': 'official',
        'model': 'resnet50',
        'train_transform': 'image_resize_and_center_crop',
        'eval_transform': 'image_resize_and_center_crop',
        'resize_scale': 256.0/224.0,
        'model_kwargs': {'pretrained': True},
        'loss_function': 'cross_entropy',
        'groupby_fields': ['background', 'y'],
        'val_metric': 'acc_wg',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'SGD',
        'optimizer_kwargs': {'momentum':0.9},
        'scheduler': None,
        'batch_size': 128,
        'lr': 1e-5,
        'weight_decay': 1.0,
        'n_epochs': 300,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'yelp': {
        'split_scheme': 'official',
        'model': 'bert-base-uncased',
        'train_transform': 'bert',
        'eval_transform': 'bert',
        'max_token_length': 512,
        'loss_function': 'cross_entropy',
        'algo_log_metric': 'accuracy',
        'batch_size': 8,
        'lr': 2e-6,
        'weight_decay': 0.01,
        'n_epochs': 3,
        'n_groups_per_batch': 2,
        'process_outputs_function': 'multiclass_logits_to_pred',
    },
    'sqf': {
        'split_scheme': 'all_race',
        'model': 'logistic_regression',
        'train_transform': None,
        'eval_transform': None,
        'model_kwargs': {'in_features': 104},
        'loss_function': 'cross_entropy',
        'groupby_fields': ['y'],
        'val_metric': 'precision_at_global_recall_all',
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'optimizer': 'Adam',
        'optimizer_kwargs': {},
        'scheduler': None,
        'batch_size': 4,
        'lr': 5e-5,
        'weight_decay': 0,
        'n_epochs': 4,
        'process_outputs_function': None,
    },
}

##########################################
### Split-specific defaults for Amazon ###
##########################################

amazon_split_defaults = {
    'official':{
        'groupby_fields': ['user'],
        'val_metric': '10th_percentile_acc',
        'val_metric_decreasing': False,
        'no_group_logging': True,
    },
    'user':{
        'groupby_fields': ['user'],
        'val_metric': '10th_percentile_acc',
        'val_metric_decreasing': False,
        'no_group_logging': True,
    },
    'time':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
    'time_baseline':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
}

user_baseline_splits = [
    'A1CNQTCRQ35IMM_baseline', 'A1NE43T0OM6NNX_baseline', 'A1UH21GLZTYYR5_baseline', 'A20EEWWSFMZ1PN_baseline',
    'A219Y76LD1VP4N_baseline', 'A37BRR2L8PX3R2_baseline', 'A3JVZY05VLMYEM_baseline', 'A9Q28YTLYREO7_baseline',
    'ASVY5XSYJ1XOE_baseline', 'AV6QDP8Q0ONK4_baseline'
    ]
for split in user_baseline_splits:
    amazon_split_defaults[split] = {
        'groupby_fields': ['user'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        }

category_splits = [
    'arts_crafts_and_sewing_generalization', 'automotive_generalization',
    'books,movies_and_tv,home_and_kitchen,electronics_generalization', 'books_generalization', 'category_subpopulation',
    'cds_and_vinyl_generalization', 'cell_phones_and_accessories_generalization', 'clothing_shoes_and_jewelry_generalization',
    'digital_music_generalization', 'electronics_generalization', 'grocery_and_gourmet_food_generalization',
    'home_and_kitchen_generalization', 'industrial_and_scientific_generalization', 'kindle_store_generalization',
    'luxury_beauty_generalization', 'movies_and_tv,books,home_and_kitchen_generalization', 'movies_and_tv,books_generalization',
    'movies_and_tv_generalization', 'musical_instruments_generalization', 'office_products_generalization',
    'patio_lawn_and_garden_generalization', 'pet_supplies_generalization', 'prime_pantry_generalization',
    'sports_and_outdoors_generalization', 'tools_and_home_improvement_generalization', 'toys_and_games_generalization',
    'video_games_generalization',
    ]
for split in category_splits:
    amazon_split_defaults[split] = {
        'groupby_fields': ['category'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
        }

########################################
### Split-specific defaults for Yelp ###
########################################

yelp_split_defaults = {
    'official':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
    'user':{
        'groupby_fields': ['user'],
        'val_metric': '10th_percentile_acc',
        'val_metric_decreasing': False,
        'no_group_logging': True,
    },
    'time':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
    'time_baseline':{
        'groupby_fields': ['year'],
        'val_metric': 'acc_avg',
        'val_metric_decreasing': False,
    },
}
########################################
### Split-specific defaults for scotus ###
########################################
# scotus_split_defaults = {
#     'official': {
#         'dataset_kwargs': {
#             'protected_attribute': 'decisionDirection'
#         }
#     },
#     'temporal':{
#         'dataset_kwargs': {
#             'protected_attribute': 'decisionDirection',
#         }
#         },
#     'uniform':{
#         'dataset_kwargs': {
#             'protected_attribute': 'issueArea',
#         }
#         }
# }
###############################
### Split-specific defaults ###
###############################

split_defaults = {
    'amazon': amazon_split_defaults,
    'yelp': yelp_split_defaults,
    # 'scotus': scotus_split_defaults
}
