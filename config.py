"""
DistilBERT微调配置文件
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "distilbert-base-uncased"
    local_model_path: str = "./models/distilbert-base-uncased"  # 本地模型路径
    use_local_model: bool = True  # 是否使用本地模型
    max_length: int = 512
    num_labels: int = 28  # GoEmotions多标签
    
    # 数据集配置
    dataset_name: str = "emotion"  # 数据集名称
    train_file: str = "data/full_dataset/goemotions_3.csv"  # 多标签pkl
    max_samples: Optional[int] = None  # 最大样本数量，None表示使用全部
    label_names: List[str] = GOEMOTIONS_LABELS
    
    # 训练配置
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.00
    
# 使用 default_factory 创建标签列表的副本
    label_names: List[str] = field(default_factory=lambda: GOEMOTIONS_LABELS.copy())
    
    # 输出配置
    output_dir: str = "outputs"
    save_steps: int = 500
    eval_steps: int = 500 
    logging_steps: int = 100
    
    # 其他配置
    seed: int = 42
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def get_model_path(self):
        if self.use_local_model and os.path.exists(self.local_model_path):
            return self.local_model_path
        else:
            return self.model_name 