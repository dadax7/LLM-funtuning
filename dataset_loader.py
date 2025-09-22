"""
开源文本分类数据集加载器（多标签支持）
"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

class MultiLabelTextClassificationDataset(Dataset):
    def __init__(self, texts, multihots, tokenizer, max_length=512):
        self.texts = texts
        self.multihots = multihots
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        multihot = self.multihots[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(multihot, dtype=torch.float)
        }

class DatasetLoader:
    def __init__(self, config):
        self.config = config
        self.datasets = {
            'custom': {
                'name': 'custom',
                'description': '自定义多标签数据集',
                'num_labels': config.num_labels,
                'label_names': config.label_names
            }
        }

    def load_multilabel_custom_dataset(self):
        # 直接使用上传的 CSV 文件路径
        df = pd.read_csv(self.config.train_file)

        # 获取文本内容
        texts = df['text'].tolist()

        # 获取多热编码数组，列为情感标签
        label_columns = self.config.label_names  # 从配置中获取标签列名
        multihots = df[label_columns].to_numpy()

        return texts, multihots

    def prepare_data(self, dataset_name, max_samples=None):
        if dataset_name == "custom" or self.config.train_file.endswith('.csv'):
            texts, multihots = self.load_multilabel_custom_dataset()
            if max_samples:
                texts = texts[:max_samples]
                multihots = multihots[:max_samples]
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                texts, multihots, test_size=0.3, random_state=self.config.seed
            )
            eval_texts, test_texts, eval_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, random_state=self.config.seed
            )
            return (train_texts, train_labels), (eval_texts, eval_labels), (test_texts, test_labels)
        raise NotImplementedError("当前仅支持多标签自定义CSV数据集")

    def create_multilabel_data_loaders(self, train, eval, test):
        model_path = self.config.get_model_path()
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        train_dataset = MultiLabelTextClassificationDataset(train[0], train[1], tokenizer, self.config.max_length)
        eval_dataset = MultiLabelTextClassificationDataset(eval[0], eval[1], tokenizer, self.config.max_length)
        test_dataset = MultiLabelTextClassificationDataset(test[0], test[1], tokenizer, self.config.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        return train_loader, eval_loader, test_loader, tokenizer 