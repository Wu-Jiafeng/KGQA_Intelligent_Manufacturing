# -*- coding:utf-8 -*-
import pickle
import sys
import os
import yaml

import numpy as np
import torch
import torch.optim as optim
from NER.data_manager import DataManager,build_corpus,build_map
from NER.model import BiLSTMCRF,CRFModel
from NER.utils import f1_score, get_tags, format_result,get_label_ch2en_dict,Metrics
from tqdm import tqdm

class CRF_NER(object):
    def __init__(self,entry="train",model_dir="./models/"):
        print('loading...')
        self.model_dir=model_dir
        self.load_config()
        if entry == "train":
            self.train_word_lists, self.train_tag_lists, self.word2id, self.tag2id = build_corpus("train")
            self.dev_word_lists, self.dev_tag_lists = build_corpus("dev", make_vocab=False)
            self.test_word_lists, self.test_tag_lists = build_corpus("test", make_vocab=False)
        elif entry == "test":
            self.test_word_lists, self.test_tag_lists = build_corpus("test", make_vocab=False)

        if os.path.exists(self.CRF_MODEL_PATH):
            self.model = self.load_model(self.CRF_MODEL_PATH)
        else:
            self.model = CRFModel()

    def train_eval(self,remove_O=False):
        # 训练CRF模型
        self.model.train(self.train_word_lists, self.train_tag_lists)
        self.save_model(self.CRF_MODEL_PATH)

        pred_tag_lists = self.model.test(self.dev_word_lists)

        metrics = Metrics(self.dev_tag_lists, pred_tag_lists, remove_O=remove_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()

        return pred_tag_lists

    def test(self):
        crf_pred = self.model.test(self.test_word_lists)
        metrics = Metrics(self.test_tag_lists, crf_pred)
        metrics.report_scores()
        metrics.report_confusion_matrix()

    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
        pred_tag_lists = self.model.test([list(input_str)])
        entities = []
        for tag in self.tags:
            tags = get_tags(pred_tag_lists[0], tag)
            entities += format_result(tags, input_str, tag)
        return entities

    def load_config(self):
        fopen = open(self.model_dir+"config.yml", "r", encoding="utf-8")
        config = yaml.load(fopen)
        fopen.close()
        self.tags,self.ch2en_dict = config.get("tags"),get_label_ch2en_dict()
        self.tags=[self.ch2en_dict[tag] for tag in self.tags]
        self.CRF_MODEL_PATH = self.model_dir+'crf.pkl'

    def load_model(self,file_name):
        """用于加载模型"""
        with open(file_name, "rb") as f:
            model = pickle.load(f)
        return model

    def save_model(self,file_name):
        """用于保存模型"""
        with open(file_name, "wb") as f:
            pickle.dump(self.model, f)

class BiLSTM_CRF_NER(object):
    
    def __init__(self, entry="train",model_dir="./models/"):
        self.model_dir=model_dir
        self.load_config()
        self.__init_model(entry)

    def __init_model(self, entry):
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)
            dev_manager = DataManager(batch_size=30, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()
        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()
        elif entry == "test":
            self.test_manager = DataManager(batch_size=self.batch_size, tags=self.tags,data_type="test")
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()

    def load_config(self):
        try:
            fopen = open(self.model_dir+"config.yml","r",encoding="utf-8")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open(self.model_dir+"config.yml", "w",encoding="utf-8")
            config = {
                "embedding_size": 100,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout":0.5,
                "model_path": "models/",
                "tags": ["ORG", "PER"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = self.model_dir
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    def restore_model(self):
        try:
            checkpoint=torch.load(self.model_path + "params.pkl")
            self.model.load_state_dict(checkpoint)
            if "best_f1" not in checkpoint.keys(): self.best_f1=0.0
            print("model restore success!")
        except Exception as error:
            print("model restore failed! {}".format(error))

    def save_params(self, data):
        with open(self.model_dir+"data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open(self.model_dir+"data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        for epoch in range(100):
            index = 0
            tbar = tqdm(self.train_manager.get_batch(),total=self.total_size, desc='Training Epoch ' + str(epoch))
            for batch in tbar:
                index += 1
                self.model.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                tbar.set_postfix(loss=loss.cpu().tolist()[0])
                loss.backward()
                optimizer.step()
            print("\n" + "-" * 50)
            f1_current=self.evaluate()
            if f1_current>=self.best_f1:
                print('\nBest checkpoint!\n')
                self.best_f1=f1_current
                torch.save(self.model.state_dict(), self.model_path+'params.pkl')
            print("-" * 50)

    def evaluate(self):
        f1_avr,tag_sum=0.,0.
        sentences, labels, length = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences)
        print("\teval")
        for tag in self.tags:
            recall, precision, f1=f1_score(labels, paths, tag, self.model.tag_map)
            f1_avr+=f1
            if f1!=0: tag_sum+=1
        return f1_avr/tag_sum

    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
        input_vec = [self.vocab.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1)
        _, paths = self.model(sentences)

        entities = []
        for tag in self.tags:
            tags = get_tags(paths[0], tag, self.tag_map)
            entities += format_result(tags, input_str, tag)
        return entities

    def test(self):
        pred,gold=[],[]
        for input_vec,label in tqdm(self.test_manager.data,desc="Testing"):
            sentence = torch.tensor(input_vec).view(1, -1)
            _, paths = self.model(sentence)
            pred.append(paths[0])
            gold.append(label)
        metrics = Metrics(gold, pred,tag_map=self.tag_map)
        metrics.report_scores()
        metrics.report_confusion_matrix()




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("menu:\n\ttrain bilstm_crf/crf\n\ttest bilstm_crf/crf\n\tpredict bilstm_crf/crf")
        exit()
    if sys.argv[1] == "train":
        if len(sys.argv)==3 and sys.argv[2]=="crf":
            print("Training CRF!")
            cn = CRF_NER("train")
            crf_pred = cn.train_eval()
        else :
            print("Training BiLSTM_CRF!")
            cn = BiLSTM_CRF_NER("train")
            cn.train()
    elif sys.argv[1] == "predict":
        if len(sys.argv)==3 and sys.argv[2]=="crf":
            print("Predicting CRF!")
            cn = CRF_NER("predict")
        else :
            print("Predicting BiLSTM_CRF!")
            cn = BiLSTM_CRF_NER("predict")
        print(cn.predict())
    elif sys.argv[1] == "test":
        if len(sys.argv)==3 and sys.argv[2]=="crf":
            print("Test CRF!")
            cn = CRF_NER("test")
        else :
            print("Test BiLSTM_CRF!")
            cn = BiLSTM_CRF_NER("test")
        print(cn.test())
