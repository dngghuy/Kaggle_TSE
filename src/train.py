import config
import dataset
import engine
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import BERTBaseUncased
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    dfx = pd.read_csv(config.TRAIN_FILE).dropna().reset_index(drop=True)
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.2,
        random_state=7664,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device('cuda')
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', "LayerNorm.weight"]
    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.
        },
    ]

    num_train_steps = int(df_train.__len__() / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=40,
        num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    best_jaccard = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, device, scheduler)
        # jaccard = engine.eval_fn(valid_dataloader, model, device)


if __name__ == '__main__':
    run()



