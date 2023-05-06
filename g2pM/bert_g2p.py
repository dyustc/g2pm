import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

SPLIT_TOKEN = "▁"


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class G2PBert(nn.Module):
    def __init__(self, num_classes):
        super(G2PBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, poly_ids):
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask}
        outputs = self.bert(**inputs)
        hidden = outputs[0]
        batch_size = input_ids.size(0)
        poly_hidden = hidden[torch.arange(batch_size), poly_ids]
        logits = self.classifier(poly_hidden)

        return logits


class G2PDataset(Dataset):
    def __init__(self, sent_file, label_file, class2idx_file, max_length=512):
        super(G2PDataset, self).__init__()
        self.max_length = max_length
        self.sents = open(sent_file).readlines()
        self.labels = open(label_file).readlines()

        assert len(self.sents) == len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        with open(class2idx_file, "rb") as f:
            self.class2idx = pickle.load(f)
        self.num_classes = len(self.class2idx)
        self.total_size = len(self.labels)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        cls_tok = "[CLS]"
        sep_tok = "[SEP]"
        sent = self.sents[index].strip()
        label = self.labels[index].strip()

        sent = sent.replace(SPLIT_TOKEN, cls_tok)
        toks = self.tokenizer.tokenize(sent)

        poly_idx = toks.index(cls_tok) + 1

        toks = list(filter(lambda x: x != cls_tok, toks))
        toks.insert(0, cls_tok)
        toks.append(sep_tok)

        input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_id = self.class2idx[label]

        return input_ids, poly_idx, label_id


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs
    all_input_ids, poly_ids, label_ids = zip(*data)

    all_input_ids = merge(all_input_ids)
    poly_ids = torch.tensor(poly_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)

    return all_input_ids, poly_ids, label_ids


def get_dataloader(sent_file, label_file, class2idx,
                   batch_size, max_length, shuffle=False):

    dataset = G2PDataset(sent_file, label_file, class2idx, max_length)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=4)

    return dataloader


def trunc_length(input_ids):
    length = torch.sum(torch.sign(input_ids), 1)
    max_length = torch.max(length)

    input_ids = input_ids[:, :max_length]

    return input_ids


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    train_dataloader = get_dataloader(args.train_file, args.train_label,
                                      args.class2idx, args.batch_size,
                                      args.max_length, shuffle=True)

    val_dataloader = get_dataloader(args.val_file, args.val_label,
                                    args.class2idx, args.batch_size,
                                    args.max_length, shuffle=True)

    test_dataloader = get_dataloader(args.test_file, args.test_label,
                                     args.class2idx, args.batch_size,
                                     args.max_length, shuffle=True)

    with open(args.class2idx, "rb") as f:
        class2idx = pickle.load(f)
    print("num classes: {}".format(len(class2idx)))
    num_classes = len(class2idx)
    model = G2PBert(num_classes)
    device = torch.cuda.current_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model_dir = "./save/bert"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader, start=1):
            batch = tuple(t.to(device) for t in batch)
            input_ids, poly_ids,  labels = batch
            mask = torch.sign(input_ids)

            inputs = {"input_ids": input_ids,
                      "poly_ids": poly_ids,
                      "attention_mask": mask}
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

            if idx % 100 == 0:
                print("loss : {:.4f}".format(loss.item()))
        all_preds = []
        all_labels = []
        model.eval()
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, poly_ids,  labels = batch
            mask = torch.sign(input_ids)

            inputs = {"input_ids": input_ids,
                      "poly_ids": poly_ids,
                      "attention_mask": mask}
            with torch.no_grad():
                logits = model(**inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        val_acc = accuracy_score(labels, preds)
        print("epoch :{}, acc: {:.2f}".format(epoch, val_acc*100))
        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = model.state_dict()
            save_file = os.path.join(
                model_dir, "{:.2f}_model.pt".format(val_acc*100))
            torch.save(state_dict, save_file)

    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, poly_ids, labels = batch
        mask = torch.sign(input_ids)
        inputs = {"input_ids": input_ids,
                  "poly_ids": poly_ids,
                  "attention_mask": mask}

        with torch.no_grad():
            logits = model(**inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)

    test_acc = accuracy_score(labels, preds)
    print("Final acc: {:.2f}".format(test_acc*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../data/train.sent")
    parser.add_argument("--train_label", type=str, default="../data/train.lb")
    parser.add_argument("--class2idx",
                        type=str,
                        default="../data/class2idx.pkl")

    parser.add_argument("--val_file", type=str, default="../data/dev.sent")
    parser.add_argument("--val_label", type=str, default="../data/dev.lb")

    parser.add_argument("--test_file", type=str, default="../data/test.sent")
    parser.add_argument("--test_label", type=str, default="../data/test.lb")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    main(args)
