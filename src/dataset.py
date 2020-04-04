import config
import torch
import numpy as np
import pandas as pd


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return self.tweet.__len__()

    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())

        len_selected_text = selected_text.__len__()
        idx0 = -1
        idx1 = -1
        selected_start = [i for i, e in enumerate(tweet) if e == selected_text[0]]
        for ind in selected_start:
            if tweet[ind: ind + len_selected_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_selected_text - 1
                break

        # Find selected words as character targets
        char_targets = [0] * tweet.__len__()
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tokenized_tweet = self.tokenizer.encode(tweet)
        tokenized_tweet_tokens = tokenized_tweet.tokens
        tokenized_tweet_ids = tokenized_tweet.ids
        tokenized_tweet_offsets = tokenized_tweet.offsets[1:-1]

        # Interesting part here, since we have words that are partially encoded, hence we use partial match rule
        targets = [0] * (tokenized_tweet_tokens.__len__() - 2)
        for j, (offset1, offset2) in enumerate(tokenized_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                targets[j] = 1

        targets = [0] + targets + [0] # Add [CLS], [SEP]
        targets_start = [0] * targets.__len__()
        targets_end = [0] * targets.__len__()

        non_zero_vars = np.nonzero(targets)[0]
        if non_zero_vars.__len__() > 0:
            targets_start[non_zero_vars[0]] = 1
            targets_end[non_zero_vars[-1]] = 1

        # Attention mask
        mask = [1] * tokenized_tweet_ids.__len__()
        token_type_ids = [0] * tokenized_tweet_ids.__len__()

        padding_len = self.max_len - tokenized_tweet_ids.__len__()
        ids = tokenized_tweet_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len

        sentiment = [1, 0, 0] # neutral
        if self.sentiment[item] == 'positive':
            sentiment = [0, 0, 1]
        if self.sentiment[item] == 'negative':
            sentiment = [0, 1, 0]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_len, dtype=torch.long),
            'tweet_tokens': " ".join(tokenized_tweet_tokens),
            'original_tweet': self.tweet[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'original_sentiment': self.sentiment[item],
            'original_selected': self.selected_text[item]
        }

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_FILE).dropna().reset_index(drop=True)
    dset = TweetDataset(
        tweet=df.text.values,
        sentiment=df.sentiment.values,
        selected_text=df.selected_text.values
    )
    print(dset[0])
