# Fairlex: A Multilingual Benchmark for Evaluating Fairness in Legal Text Processing

This repository is an extension of the [WILDS](https://github.com/p-lambda/wilds) library for the FairLex benchmark. 

The scope of this work is to provide an evaluation framework along with extensive experiments to further study fairness within the legal domain. Following the work of Angwin et al., (2016), Dressel et al. (2018), and Wang et al. (2021), we provide a diverse benchmark covering multiple tasks, jurisdictions, and protected (examined) attributes.
We conduct experiments based on state-of-the-art pre-trained transformer-based language models and compare model performance across four representative group-robust algorithm, i.e., Adversarial Removal (Elazar and Goldberg, 2018), Group DRO (Sagawa et al., 2020), IRM (Arjovsky et al., 2020) and REx (Krueger et al., 2020).

We believe that this work can help practitioners to build assisting technology for legal professionals - with respect to the legal framework (jurisdiction) they operate -; technology that does not only rely on performance on majority groups, but also considering minorities and the robustness of the developed models across them. We believe that this is an important application field, where more research should be conducted (Tsarapatsanis And Aletras, 2021) in order to improve legal services and democratize law, but more importantly highlight (inform the audience on) the various multi-aspect shortcomings seeking a responsible and ethical (fair) deployment of technology.

## Citation Information

[*Ilias Chalkidis, Tommaso Pasini, Sheng Zhang, Letizia Tomada, Letizia, Sebastian Felix Schwemer, Anders Søgaard.*
*FairLex: A Multilingual Benchmark for Evaluating Fairness in Legal Text Processing.*
*2022. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, Dublin, Ireland.*](https://arxiv.org/abs/xxx/xxx)
```
@inproceedings{chalkidis-etal-2022-fairlex,
      author={Chalkidis, Ilias and Passini, Tommaso and Zhang, Sheng and
              Tomada, Letizia and Schwemer, Sebastian Felix and Søgaard, Anders},
      title={FairLex: A Multilingual Benchmark for Evaluating Fairness in Legal Text Processing},
      booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
      year={2022},
      address={Dublin, Ireland}
}
```

## Dataset

### Dataset Summary

We present a benchmark suite of four datasets for evaluating the fairness of pre-trained legal language models and the techniques used to fine-tune them for downstream tasks. Our benchmarks cover four jurisdictions (European Council, USA, Swiss, and Chinese), five languages (English, German, French, Italian and Chinese) and fairness across five attributes (gender, age, nationality/region, language, and legal area). In our experiments, we evaluate pre-trained language models using several group-robust fine-tuning techniques and show that performance group disparities are vibrant in many cases, while none of these techniques guarantee fairness, nor consistently mitigate group disparities. Furthermore, we provide a quantitative and qualitative analysis of our results, highlighting open challenges in the development of robustness methods in legal NLP.

### Dataset Repository

The dataset is available on [Hugging Face datasets](https://huggingface.co/datasets/coastalcph/fairlex) and you can easily load any dataset. For example for ECtHR dataset:

```python
from datasets import load_dataset

dataset = load_dataset("coastalcph/fairlex", "ecthr")
# you can use any of the following config names as a second argument:
"ecthr", "scotus", "fscs", "cail"
```

Note: You don't need to download or install any dataset manually, the code is doing that automatically.

## Installation Requirements

```
torch>=1.9.0
transformers>=4.8.1
requests>=2.25.1
wilds>=1.2.0
scikit-learn>=0.24.1
tqdm>=4.61.1
numpy>=1.20.1
pandas>=1.2.4
pytorch_revgrad>=0.2.0
datasets>=1.17.0
```

We strongly recommend to you use Anaconda to set a clean environment for this project.

## Configuration
The code and the configuration for `Fairlex` datasets (`ecthr`, `scotus`, `fscs`, `cail`) follow the WILDS framework.

All configurations for the datasets are available in `configs/datasets.py`

For example, the configuration for FSCS:

```python
dataset_defaults = {
    'fscs': {
        'split_scheme': 'official',
        'model': 'fairlex-fscs-minilm',
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
        'n_epochs': 20,
        'n_groups_per_batch': 2,
        'groupby_fields': ['defendant_state'],
        'irm_lambda': 0.5,
        'coral_penalty_weight': 0.5,
        'adv_lambda': 0.5,
        'rex_beta': 0.5,
        'val_metric_decreasing': False,
        'loader_kwargs': {
            'num_workers': 0,
            'pin_memory': True,
        }
    }
}
```

Note that `configs/supported.py` and `configs/model.py` also have corresponding modifacation compared to original code. 

## Models

For the purpose of this work, we release four domain-specific BERT models with continued pre-training on the corpora of the examined datasets (ECtHR, SCOTUS, FSCS, CAIL). We train mini-sized BERT models with 6 Transformer blocks, 384 hidden units, and 12 attention heads. We warm-start all models from the public MiniLMv2 (Wang et al., 2021) using the distilled version of RoBERTa (Liu et al., 2019). For the English datasets (ECtHR, SCOTUS) and the one distilled from XLM-R (Conneau et al., 2021) for the rest (trilingual FSCS, and Chinese CAIL). 

The models are also available from the [Hugging Face Hub](https://huggingface.co/models?search=fairlex).

### Models list

The code uses the respective models for each FairLex dataset.

| Model name                        | Dataset | Language           |
|-----------------------------------|---------|--------------------|
| `coastalcph/fairlex-ecthr-minlm`  | ECtHR   | `en`               |
| `coastalcph/fairlex-scotus-minlm` | SCOTUS  | `en`               |
| `coastalcph/fairlex-fscs-minlm`   | FSCS    | [`de`, `fr`, `it`] |
| `coastalcph/fairlex-cail-minlm`   | CAIL    | `zh`               |



## Run Experiments

You can use the following command to run the code:

```bash
sh scripts/run_algo.sh
```

You have to update the parameters to use the dataset, algorithm, and group field of your interest. The available parameters are:

| Dataset           | Loss Function          | Group Field        | N Groups | Missing Values |
|-------------------|------------------------|--------------------|----------|----------------|
| `ecthr`           | `binary_cross_entropy` |  `applicant_gender` | 3        | Yes            |
|                   |                        | `applicant_age`     | 4                   | Yes            |
|                   |                        | `defendant_state`   | 2                   | No             |
| `scotus`          | `cross_entropy`        |`respondent_type`   | 6                   | Yes            |
|   |                        | `decision_direction` | 2                  | No             |
| `fscs`            | `cross_entropy`        |`language`          | 3                   | No             | 
| |                        | `legal_area`         | 6                  | Yes            |
|   |                        | `court_region`       | 9                  | No             |
| `cail`            | `cross_entropy`        |`applicant_gender`  | 2                   | No             |
|   |                        | `court_region`       | 7                   | No             |


For example to run experiments for ECtHR with the ERM algorithm with data grouped by applicant's gender:

```bash
ALGORITHM='ERM'
DATASET='ecthr'
GROUP_FIELD=applicant_gender
BATCH_SIZE=16
WANDB_PROJECT=fairlex-wilds
LOG_DIR=ecthr_logs
N_GROUPS=3
LOSS_FUNCTION=cross_entropy
FOLDER_NAME='ERM'
```

## Comments

Feel free to leave comments or any contribution to the repository.

Please contact ilias.chalkidis@di_dot_ku_dot_dk if you have any concern.
