"""
DistilBERT多标签微调主脚本
"""
import torch
import random
import numpy as np
import os
from config import TrainingConfig
from dataset_loader import DatasetLoader
from model import DistilBertForMultiLabelClassification, MultiLabelTrainer
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    print("开始DistilBERT多标签微调...")
    config = TrainingConfig()
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    dataset_loader = DatasetLoader(config)
    print(f"\n准备多标签数据集: {config.train_file}")
    train, eval, test = dataset_loader.prepare_data(config.dataset_name, config.max_samples)
    train_loader, eval_loader, test_loader, tokenizer = dataset_loader.create_multilabel_data_loaders(train, eval, test)
    print(f"\n初始化模型...")
    print(f"标签数量: {config.num_labels}")
    print(f"标签名称: {config.label_names}")
    model = DistilBertForMultiLabelClassification(
        model_path=config.get_model_path(),
        num_labels=config.num_labels
    )
    trainer = MultiLabelTrainer(model, config, device)
    print("\n开始训练...")
    trainer.train(train_loader, eval_loader)
    print("\n绘制训练历史...")
    trainer.plot_training_history()
    print("\n在测试集上评估...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"测试集结果:")
    print(f"  损失: {test_metrics['loss']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    tokenizer.save_pretrained(config.output_dir)
    print(f"\n模型和tokenizer已保存到: {config.output_dir}")
    # print("\n演示预测功能...")
    # test_texts = [
    #     "I am so happy and excited!",
    #     "This is disgusting and makes me angry.",
    #     "I feel sad and disappointed.",
    #     "Thank you so much, I feel gratitude and joy!"
    # ]
    # for text in test_texts:
    #     pred_indices, probs = trainer.predict(text, tokenizer)
    #     labels = [config.label_names[i] for i in pred_indices]
    #     print(f"文本: {text}")
    #     print(f"预测标签: {labels}")
    #     print("概率:", [f"{probs[i]:.3f}" for i in pred_indices])
    #     print()

if __name__ == "__main__":
    main() 