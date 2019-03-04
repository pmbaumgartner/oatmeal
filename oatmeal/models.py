from typing import Union

import numpy as np
import torch
from numpy import array
from pytorch_pretrained_bert.modeling import (
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
)
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):  # type: ignore
    """Make a good docstring!"""

    def __init__(self, config, num_labels=2):  # type: ignore
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(  # type: ignore
        self, input_ids, token_type_ids=None, attention_mask=None, labels=None
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels).float(),
                labels.view(-1, self.num_labels).float(),
            )
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):  # type: ignore
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):  # type: ignore
        for param in self.bert.parameters():
            param.requires_grad = True


def get_bert_binary_model() -> BertForSequenceClassification:
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    if n_gpu > 1:
        bert_model = torch.nn.DataParallel(bert_model)

    return bert_model


def get_bert_multiclass_model(num_labels: int) -> BertForSequenceClassification:
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
    if n_gpu > 1:
        bert_model = torch.nn.DataParallel(bert_model)

    return bert_model


def get_bert_multilabel_model(
    num_labels: int
) -> BertForMultiLabelSequenceClassification:
    bert_model = BertForMultiLabelSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
    if n_gpu > 1:
        bert_model = torch.nn.DataParallel(bert_model)
    return bert_model


bert_model_types = Union[
    BertForSequenceClassification, BertForMultiLabelSequenceClassification
]


def get_bert_opt(
    model: bert_model_types,
    n_train_examples: int,
    train_batch_size: int,
    num_train_epochs: int,
) -> BertAdam:
    param_opt = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [p for n, p in param_opt if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_opt if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    num_train_optimization_steps = (
        int(n_train_examples / train_batch_size) * num_train_epochs
    )
    opt = BertAdam(
        opt_grouped_parameters,
        lr=2e-5,
        warmup=0.1,
        t_total=num_train_optimization_steps,
    )
    return opt


def run_model_training(
    model: bert_model_types, opt: BertAdam, dataloader: DataLoader, epochs: int
) -> bert_model_types:
    model.to(device)
    model.train()
    for _ in trange(epochs, desc="EPOCH"):
        for batch in tqdm(dataloader, desc="ITERATION"):
            batch = tuple(t.to(device) for t in batch)
            x0, x1, x2, y = batch
            loss = model(x0, x1, x2, y)
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

    return model


def run_prediction_softmax(model: bert_model_types, dataloader: DataLoader) -> array:
    model.to(device)
    model.eval()
    all_logits = None
    for batch in tqdm(dataloader, desc="BATCH"):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            x0, x1, x2 = batch
            logits = model(x0, x1, x2)

        if all_logits is None:
            all_logits = logits.softmax(1).detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, logits.softmax(1).detach().cpu().numpy()), axis=0
            )

    return all_logits


def run_prediction_sigmoid(model: bert_model_types, dataloader: DataLoader) -> array:
    model.to(device)
    model.eval()
    all_logits = None
    for batch in tqdm(dataloader, desc="BATCH"):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            x0, x1, x2 = batch
            logits = model(x0, x1, x2)
        if all_logits is None:
            all_logits = logits.sigmoid().detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, logits.sigmoid().detach().cpu().numpy()), axis=0
            )

    return all_logits
