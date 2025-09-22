"""
DistilBERT多标签分类模型
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class DistilBertForMultiLabelClassification(nn.Module):
    def __init__(self, model_path, num_labels, dropout=0.1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # 只有在训练时（labels不为None）才计算损失
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
        # 推理时只返回logits
            return logits
        # 使用 nn.BCEWithLogitsLoss 来计算损失
        # loss_fn = nn.BCEWithLogitsLoss()
        # loss = loss_fn(logits, labels)
        # return loss, logits

class MultiLabelTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=config.learning_rate,
        #     weight_decay=config.weight_decay
        # )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config.warmup_steps
        )
        self.train_losses = []
        self.eval_losses = []
        self.eval_f1s = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="训练中"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            loss, _ = self.model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, eval_loader, threshold=0.5):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="评估中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss, logits = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > threshold).astype(int)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        return {
            'loss': total_loss / len(eval_loader),
            'f1': f1
        }

    def train(self, train_loader, eval_loader):
        best_f1 = 0
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            eval_metrics = self.evaluate(eval_loader)
            self.eval_losses.append(eval_metrics['loss'])
            self.eval_f1s.append(eval_metrics['f1'])
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {eval_metrics['loss']:.4f}")
            print(f"验证F1: {eval_metrics['f1']:.4f}")
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                torch.save(self.model.state_dict(), f"{self.config.output_dir}/best_model.pth")
                print(f"保存最佳模型，F1: {best_f1:.4f}")

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.train_losses, label='traning_loss')
        ax1.plot(self.eval_losses, label='val_loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.eval_f1s, label='valF1')
        ax2.set_title('valF1')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.show()

    def predict(self, text, tokenizer, threshold=0.5):
        self.model.eval()
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pred_indices = np.where(probs > threshold)[0]
        return pred_indices, probs 