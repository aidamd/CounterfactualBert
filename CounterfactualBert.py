#import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import pandas as pd
from utils import *
from random import *
import numpy as np
import os
import time

import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class CounterfactualBert():
    def __init__(self, params, train="", counter="",
                 corpus=""):
        for key in params:
            setattr(self, key, params[key])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, return_dict=True)
        self.preprocess_train(train, counter)

        self.model_path = os.path.join(corpus + "_saved_model", self.type, self.type)

    def preprocess_train(self, train, counter):
        train = preprocess(train)
        counter = preprocess(counter)

        self.train, self.test = dict(), dict()

        self.train["text"] = train["text"].values.tolist()
        self.train["ids"] = train["Tweet ID"].values.tolist()
        self.train["labels"] = train["hate"].values.tolist()
        self.train["perplex"] = {self.train["ids"][i]: train["perplexity"].tolist()[i]
                                 for i in range(train.shape[0])}

        """
        ### TODO: move padding to batch making
        self.train["tokens"] = self.tokenizer.tokenize(self.train["text"],
                                                       return_tensors='pt',
                                                       padding=True,
                                                       truncation=True)
        """

        self.counter = dict()
        for name, group in counter.groupby(["Tweet ID"]):
            if name in self.train["perplex"]:
                counter = self.asymmetrics(name, group,
                                           self.train["perplex"][name],
                                           self.train["labels"][self.train["ids"].index(name)])
                if not isinstance(counter, str):
                    self.counter[name] = counter.reset_index()["text"].tolist()

        # self.hate_weights = [1 - Counter(self.train["labels"])[i] / len(self.train["labels"])
        #        for i in [0, 1]]
        self.hate_weights = [1, 5]

    def preprocess_test(self, test):
        test = preprocess(test)

        self.test = dict()
        self.test["text"] = test["text"].values.tolist()
        self.test["ids"] = test["Tweet ID"].values.tolist()
        # self.test["labels"] = test["hate"].values.tolist() if "hate"
        # self.test["perplex"] = test["perplexity"].values.tolist()
        #self.test["tokens"] = tokens_to_ids(self.test["text"], self.vocab)
        return test

    def asymmetrics(self, tweet, counters, perplex, hate):
        if "asym" in self.type:
            """ based on EMNLP method, for choosing the most similar counterfactuals
            diffs = [abs(perplex - counters["perplexity"].tolist()[i])
                     for i in range(counters.shape[0])]
            diffs.sort()
            thresh = np.argmax([diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)])
            return counters.iloc[[i for i in range(counters.shape[0]) if
                                  abs(perplex - counters["perplexity"].tolist()[i]) <= diffs[thresh]]]
            """

            """ based on NAACL method, for choosing the top counterfactuals """
            return counters.iloc[[i for i in range(counters.shape[0]) if
                                  counters["perplexity"].tolist()[i] >= perplex]]

        elif "clp" in self.type:
            return "" if hate else counters
        elif "group" in self.type:

            return
        else:
            return ""

    def CV(self, folds=5):
        kfold = StratifiedKFold(n_splits=folds, shuffle=True)
        results = list()

        i = 1
        for t_idx, v_idx in kfold.split(np.arange(len(self.train["text"])),
                                        self.train["labels"]):
            print("CV:", i)
            i += 1
            t_text, v_text = [self.train["text"][t] for t in t_idx], \
                                 [self.train["text"][t] for t in v_idx]
            t_index, v_index = [self.train["ids"][t] for t in t_idx], \
                               [self.train["ids"][t] for t in v_idx]
            t_labels, v_labels = [self.train["labels"][t] for t in t_idx], \
                                 [self.train["labels"][t] for t in v_idx]
            train_batches = get_batches(t_text,
                                        t_index,
                                        self.batch_size,
                                        self.tokenizer,
                                        hate=t_labels,
                                        counter=self.counter)
            val_batches = get_batches(v_text,
                                      v_index,
                                      self.batch_size,
                                      self.tokenizer,
                                      hate=v_labels,
                                      counter=self.counter)
            results.append(self.train_model(train_batches, val_batches))
        
        print("Overall performance:")
        print("Accuracy:", sum([res["accuracy"] for res in results]) / folds,
                np.std([res["accuracy"] for res in results]))
        print("F1:", sum([res["f1"] for res in results]) / folds,
                np.std([res["f1"] for res in results]))
        print("Precision:", sum([res["precision"] for res in results]) / folds,
                np.std([res["precision"] for res in results]))
        print("Recall:", sum([res["recall"] for res in results]) / folds,
                np.std([res["recall"] for res in results]))

        train_batches = get_batches(self.train["text"],
                                    self.train["ids"],
                                    self.batch_size,
                                    self.tokenizer,
                                    hate=self.train["labels"],
                                    counter=self.counter)
        self.train_model(train_batches, [], save=True)

    def feed_dict(self, batch):
        X = [b["input"] for b in batch]
        X_att = [b["attention"] for b in batch]
        counter_idx = [randrange(len(b["counter_input"]) - 1) 
                if len(b["counter_input"]) > 1 else 0 for b in batch]
        counter = [b["counter_input"][counter_idx[i]] for i, b in enumerate(batch)]
        counter_att = [b["counter_attention"][counter_idx[i]] for i, b in enumerate(batch)]
        y = [b["hate"] for b in batch]
        return X, X_att, counter, counter_att, y

    def train_model(self, train_batches, val_batches, save=False):

        self.device = "cuda"
        self.model = self.model.to(self.device)
        #self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)
        self.hate_weights = torch.tensor([1, 5], dtype=torch.float32).to(self.device)
        self.weighted_loss = torch.nn.CrossEntropyLoss(weight=self.hate_weights)

        for epoch in range(3):
            start = time.time()
            sum_loss = 0
            self.model.train()
            # print("Epoch", epoch)
            for batch in train_batches:
                """ get X and counter """
                X, X_att, counter, counter_att, y = self.feed_dict(batch)
                
                #print(len(X[0]))
                X_ids = torch.tensor(X).to(self.device)
                X_att = torch.tensor(X_att).to(self.device)
                #counter_ids = torch.tensor(counter).to(self.device)
                #counter_att = torch.tensor(counter_att).to(self.device)
                labels = torch.tensor(y).unsqueeze(0).to(self.device)
                
                ## getting outputs for sentences
                X_outputs = self.model(X_ids, attention_mask=X_att, labels=labels)
                
                X_ids = torch.tensor(counter).to(self.device)
                X_att = torch.tensor(counter_att).to(self.device)
                ## getting outputs for counterfactuals
                counter_outputs = self.model(X_ids, attention_mask=X_att,  labels=labels)
                
                self.X_logits = X_outputs.logits
                self.counter_logits = counter_outputs.logits

                class_loss = self.weighted_loss(X_outputs.logits, torch.reshape(labels, (-1, )))
                counter_loss = torch.mean(torch.abs(self.X_logits.sub(self.counter_logits)))
                self.loss = class_loss #+ counter_loss
                self.loss.backward()
                self.optimizer.step() 
            print("Epoch", epoch, time.time() - start)
            if val_batches:
                out, res = self.predict(val_batches, self.model)
                print(res)
        if save:
            self.model.save_pretrained(self.model_path)

        if val_batches:
            #out, res = self.predict(val_batches, self.model)
            #print(res)
            return res
        else:
            return ""

    def test_model(self, test):
        test = self.preprocess_test(test)
        test = test.reset_index()
        batches = get_batches(self.test["tokens"],
                              self.test["ids"],
                              self.batch_size,
                              self.tokenizer)

        test_predictions, results = self.predict(batches)
        test["predict"] = pd.Series(test_predictions["prediction"])
        test["logits"] = pd.Series(test_predictions["logits"])
        return test, results

    def predict(self, batches, model=None):
        if not model:
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2, return_dict=True)
        self.model.eval()
        self.device = "cuda"
        self.model = self.model.to(self.device)

        outputs = {"prediction": list(),
                   "logits": list(),
                   "labels": list()}

        for batch in batches:
            X, X_att, _, _, y = self.feed_dict(batch)
            X_ids = torch.tensor(X).to(self.device)
            X_att = torch.tensor(X_att).to(self.device)
            labels = torch.tensor(y).unsqueeze(0).to(self.device)

            self.pred = self.model(X_ids, attention_mask=X_att, labels=labels)
            preds = torch.argmax(self.pred.logits, dim=-1)
            logits = self.pred.logits

            outputs["prediction"].extend([p.tolist() for p in preds])
            outputs["logits"].extend([l.tolist() for l in logits])
            outputs["labels"].extend(list(y))

            #print(outputs)

        precision, recall, f1, _ = precision_recall_fscore_support(outputs["labels"], outputs["prediction"], average='binary')   
        acc = accuracy_score(outputs["labels"], outputs["prediction"])

        results = {'accuracy': round(acc, 4),
                    'f1': round(f1, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4)}
        return outputs, results
