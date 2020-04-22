import config
import transformers
import torch
import torch.nn as nn
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        m_std = math.sqrt(2. / m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0, std=m_std)
        if m.bias is not None:
            m.bias.data.fill_(0.)


class TweetModelElectraHeader1(transformers.BertPreTrainedModel):
    def __init__(self, conf, init_type):
        super(TweetModelElectraHeader1, self).__init__(conf)
        init_func = init_dict.get(init_type, None)
        self.electra = transformers.ElectraModel.from_pretrained(config.ELECTRA_PATH, config=conf)
        self.header = nn.Sequential(
            nn.Linear(768 * 2, 2)
        )
        if init_func is not None:
            self.header.apply(init_func)

    def forward(self, ids, mask, token_type_ids):
        _, out = self.electra(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        logits = self.header(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class TweetModelElectraHeader2(transformers.BertPreTrainedModel):
    def __init__(self, conf, init_type):
        super(TweetModelElectraHeader2, self).__init__(conf)
        init_func = init_dict.get(init_type, None)
        self.electra = transformers.ElectraModel.from_pretrained(config.ELECTRA_PATH, config=conf)
        self.header = nn.Sequential(
            nn.Linear(768 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        if init_func is not None:
            self.header.apply(init_func)

    def forward(self, ids, mask, token_type_ids):
        _, out = self.electra(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        logits = self.header(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


model_dict = {
    'electra_header1': TweetModelElectraHeader1,
    'electra_header2': TweetModelElectraHeader2
}

init_dict = {
    'he': weights_init,
}