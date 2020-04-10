import utils
import torch
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import wandb
wandb.init("Tweet_Sentiment_Extraction")


def loss_func(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)

    return l1 + l2


# TODO: Using Pytorch lightning
def train_fn(data_loader, model, optimizer, device, scheduler):
    wandb.watch(model, log='all')
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=data_loader.__len__())
    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs1, outputs2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_func(outputs1, outputs2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        wandb.log({"train_loss": loss})


def eval_fn(data_loader, model, device):
    model.eval() # Eval mode
    fin_output_start = []
    fin_output_end = []
    fin_padding_len = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_tweet = []
    fin_orig_selected = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=data_loader.__len__()):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            tweet_tokens = d["tweet_tokens"]
            padding_len = d["padding_len"]
            ori_sentiment = d["original_sentiment"]
            orig_selected = d["original_selected"]
            orig_tweet = d["original_tweet"]
            targets_start = d['targets_start']
            targets_end = d['targets_end']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs1, outputs2 = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_func(outputs1, outputs2, targets_start, targets_end)
            wandb.log({'valid_loss': loss})

            fin_output_start.append(torch.sigmoid(outputs1).cpu().detach().numpy())
            fin_output_end.append(torch.sigmoid(outputs2).cpu().detach().numpy())
            fin_padding_len.extend(padding_len.cpu().detach().numpy().tolist())
            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_selected.extend(orig_selected)
            fin_orig_sentiment.extend(ori_sentiment)
            fin_orig_tweet.extend(orig_tweet)

        fin_output_start = np.vstack(fin_output_start)
        fin_output_end = np.vstack(fin_output_end)

        thresh = .2
        jaccards = []
        for j in range(len(fin_tweet_tokens)):
            target_string = fin_orig_selected[j]
            tweet_tokens = fin_tweet_tokens[j]
            padding_len = fin_padding_len[j]
            original_tweet = fin_orig_tweet[j]
            sentiment = fin_orig_sentiment[j]

            if padding_len > 0:
                mask_start = fin_output_start[j, :][:-padding_len] >= thresh
                mask_end = fin_output_end[j, :][:-padding_len] >= thresh
            else:
                mask_start = fin_output_start[j, :] >= thresh
                mask_end = fin_output_end[j, :] >= thresh

            mask = [0] * len(mask_start)
            idx_start = np.nonzero(mask_start)[0]
            idx_end = np.nonzero(mask_end)[0]

            if idx_start.__len__() > 0:
                idx_start = idx_start[0]
                if idx_end.__len__() > 0:
                    idx_end = idx_end[0]
                else:
                    idx_end = idx_start
            else:
                idx_start = 0
                idx_end = 0

            for mj in range(idx_start, idx_end + 1):
                mask[mj] = 1

            output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
            output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

            # TODO: Own rule
            final_output = ""
            for ot in output_tokens:
                if ot.startswith('##'):
                    final_output = final_output + ot[2:]
                elif ot.__len__() == 1 and ot in string.punctuation:
                    final_output = final_output + ot
                else:
                    final_output = final_output + " " + ot

            final_output = final_output.strip()
            if sentiment == 'neutral' or (original_tweet.split()).__len__() < 4:
                final_output = original_tweet
            jac = utils.jaccard(target_string.strip(), final_output.strip())
            jaccards.append(jac)
        mean_jac = np.mean(jaccards)

        return mean_jac
