import config
import dataset
import engine
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import model
import transformers
import wandb
import experiments
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim import SGD, RMSprop
from torch.optim.lr_scheduler import OneCycleLR
import utils


def random_seed(seed_value):
    """
    Define random seed for reproducible results
    """
    import random
    random.seed(seed_value) # Python
    import numpy as np
    np.random.seed(seed_value) # cpu vars
    import torch
    torch.manual_seed(seed_value) # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


random_seed(7664)


def training_adamW_scheme(optimizer_parameters, train_data_loader_length):
    """
    Setting optimizer and scheduler in AdamW scheme
    """
    num_train_steps = int(train_data_loader_length / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=40,
        num_training_steps=num_train_steps
    )

    return optimizer, scheduler


def training_SGD_scheme(optimizer_parameters, train_data_loader_length, stage=None):
    """
    Setting optimizer and scheduler in SGD scheme
    """
    optimizer = SGD(optimizer_parameters, lr=3e-3, momentum=0.85)
    if stage is None:
        stage = 1
    if stage == 1:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[4e-3, 1e-4, 1e-4],
            epochs=config.EPOCHS,
            steps_per_epoch=train_data_loader_length,
            base_momentum=[0.85, 0.85, 0.85],
            max_momentum=[0.95, 0.95, 0.95],
            pct_start=.4,
            div_factor=15,
            final_div_factor=1e4
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[2e-4, 3e-5, 3e-5],
            epochs=config.EPOCHS,
            steps_per_epoch=train_data_loader_length,
            base_momentum=[0.85, 0.85, 0.85],
            max_momentum=[0.95, 0.95, 0.95],
            pct_start=.4,
            div_factor=15,
            final_div_factor=1e4
        )

    return optimizer, scheduler


def prepare_experiment_dataset():
    """
    Preparing train loader and valid loader from default train and valid csv for reliable comparisons.
    """
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_valid=  pd.read_csv(config.VALID_FILE)

    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    return train_data_loader, valid_data_loader


def get_experiment_model(model_type, init_type, device):
    model_config = transformers.ElectraConfig.from_pretrained(config.ELECTRA_PATH)
    model_config.output_hidden_states = True
    model_callable = model.model_dict.get(model_type)
    experiment_model = model_callable(model_config, init_type)
    experiment_model.to(device)

    return experiment_model


def run_experiments_adamw(experiment_config, wandb_name=None, wandb_notes=None):
    """
    """
    if wandb_name is not None:
        wandb.init(project='TSE_Electra_Experiments',
                   reinit=True,
                   name=wandb_name)
    else:
        wandb.init(project='TSE_Electra_Experiments',
                   reinit=True,
                   name='temporary_run_AdamW')

    model_type = experiment_config.get('model')
    init_type = experiment_config.get('init')
    # Default, since I am running on Google Colab
    device = torch.device("cuda")
    MODELS = config.MODEL_PATH / 'AdamW' / model_type / init_type
    utils.check_make_dir(MODELS)

    train_data_loader, valid_data_loader = prepare_experiment_dataset()

    experiment_model = get_experiment_model(model_type, init_type, device)
    param_optimizer = list(experiment_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not ('bert' in n)],
            'weight_decay': 0.001,
        },
        {
            'params': [p for n, p in param_optimizer if ('bert' in n) and not any(nd in n for nd in no_decay)],
            'weight_decay': 0.001,
        },
        {
            'params': [p for n, p in param_optimizer if ('bert' in n) and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    optimizer, scheduler = training_adamW_scheme(optimizer_parameters, train_data_loader.__len__())

    es = utils.EarlyStopping(patience=3, mode="max")

    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, experiment_model, optimizer, device, scheduler=scheduler, wandb_notes=wandb_notes)
        jaccard = engine.eval_fn(valid_data_loader, experiment_model, device, wandb_notes=wandb_notes)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"{str(MODELS)}/AdamW_{model_type}_{init_type}.bin")
        if es.early_stop:
            print("Early stopping")
            break


def run_experiments_sgd(experiment_config, stage=1, wandb_name=None, wandb_notes=None):
    """
    """
    if wandb_name is not None:
        wandb.init(project='TSE_Electra_Experiments',
                   reinit=True,
                   name=wandb_name)
    else:
        wandb.init(project='TSE_Electra_Experiments',
                   reinit=True,
                   name='temporary_run_SGD')

    model_type = experiment_config.get('model')
    init_type = experiment_config.get('init')
    # Default, since I am running on Google Colab
    device = torch.device("cuda")
    MODELS = config.MODEL_PATH / 'SGD' / model_type / init_type
    utils.check_make_dir(MODELS)

    train_data_loader, valid_data_loader = prepare_experiment_dataset()

    experiment_model = get_experiment_model(model_type, init_type, device)
    param_optimizer = list(experiment_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not ('bert' in n)],
            'weight_decay': 0.001,
        },
        {
            'params': [p for n, p in param_optimizer if ('bert' in n) and not any(nd in n for nd in no_decay)],
            'weight_decay': 0.001,
        },
        {
            'params': [p for n, p in param_optimizer if ('bert' in n) and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    optimizer, scheduler = training_SGD_scheme(optimizer_parameters, train_data_loader.__len__(), stage=stage)

    es = utils.EarlyStopping(patience=3, mode="max")

    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, experiment_model, optimizer, device, scheduler=scheduler, wandb_notes=wandb_notes)
        jaccard = engine.eval_fn(valid_data_loader, experiment_model, device, wandb_notes=wandb_notes)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"{str(MODELS)}/SGD_{model_type}_{init_type}.bin")
        if es.early_stop:
            print("Early stopping")
            break

if __name__ == '__main__':
    experiment1 = experiments.experiment1
    experiment2 = experiments.experiment2

    run_experiments_adamw(experiment1, wandb_name='AdamW_experiment1', wandb_notes='stage1')
    run_experiments_adamw(experiment2, wandb_name='AdamW_experiment2', wandb_notes='stage1')

