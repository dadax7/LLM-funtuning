"""
推理脚本 - 使用训练好的DistilBERT模型进行预测
"""
import torch
import numpy as np
from transformers import DistilBertTokenizer
from model import DistilBertForMultiLabelClassification
from config import TrainingConfig

class DistilBertInference:
    """DistilBERT推理类"""
    
    def __init__(self, model_path, config_path=None):
        self.config = TrainingConfig() if config_path is None else TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 获取模型路径
        config_model_path = self.config.get_model_path()
        
        # 加载tokenizer
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(config_model_path)
            print(f"✅ 成功加载本地tokenizer: {config_model_path}")
        except Exception as e:
            print(f"⚠️ 本地tokenizer加载失败: {e}")
            print("使用在线tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        
        # 初始化模型
        self.model = DistilBertForMultiLabelClassification(
            model_path=config_model_path,
            num_labels=self.config.num_labels
        )
        
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"标签数量: {self.config.num_labels}")
        print(f"标签名称: {self.config.label_names}")
    
    def predict(self, text):
        """预测单个文本"""
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
        
        return predictions.cpu().numpy()[0], probabilities.cpu().numpy()[0]
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        
        for text in texts:
            prediction, probabilities = self.predict(text)
            results.append({
                'text': text,
                'prediction': prediction,
                'probabilities': probabilities
            })
        
        return results
    
    def predict_with_confidence(self, text, confidence_threshold=0.5):
        """带置信度阈值的预测"""
        prediction, probabilities = self.predict(text)
        confidence = probabilities[prediction]
        

        
        label_name = self.config.label_names[prediction]
        return label_name, confidence, probabilities

def main():
    """主函数 - 演示推理功能"""
    # 模型路径
    model_path = "outputs/best_model_1e-4_adam_0.2949_data2.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    # 初始化推理器
    inference = DistilBertInference(model_path)
    
    # 根据数据集类型选择测试文本
    config = TrainingConfig()
    
    if config.dataset_name == 'ag_news':
        test_texts = [
            "The stock market reached new highs today.",
            "The team won the championship game.",
            "Scientists discovered a new species.",
            "The company announced quarterly earnings."
        ]
    elif config.dataset_name == 'sst2':
        test_texts = [
            "This movie is really great, I love it!",
            "The service is terrible, I won't come back.",
            "Today the weather is nice, I'm happy."
        ]
    elif config.dataset_name == 'emotion':
        test_texts = [
            "I'm so happy to see you again!",
            "This makes me really angry.",
            "I like you!",
            "I don't like it.",
            "I'm scared of the dark."
            "The stock market reached new highs today.",
            "The team won the championship game.",
            "Scientists discovered a new species.",
            "The company announced quarterly earnings."
            "This movie is really great, I love it!",
            "The service is terrible, I won't come back.",
            "Today the weather is nice, I'm happy.",
            "This movie is really good, I like it very much!" ,
"This product is so bad that I don't recommend it at all." ,
"It's a fine day and a cheerful mood today." ,
"The service was very bad. Won't come again." ,
"The food in this restaurant is delicious and the environment is nice." ,
"I'm disappointed that the product is of poor quality." ,
"The teacher was very lively and learned a lot." ,
"This software is too difficult to use. The interface is not friendly."
        ]
    else:
        # 默认测试文本
        test_texts = [
            "这部电影真的很棒，我非常喜欢！",
            "这个产品太差了，完全不推荐。",
            "今天天气很好，心情愉快。",
            "服务态度很差，不会再来了。",
            "这个餐厅的菜很好吃，环境也不错。",
            "产品质量有问题，很失望。",
            "老师讲课很生动，学到了很多。",
            "这个软件太难用了，界面不友好。"
        ]
    
    print("开始推理...")
    print("=" * 50)
    
    # 单个预测
    print("单个预测示例:")
    for text in test_texts[:3]:
        sentiment, confidence, probabilities = inference.predict_with_confidence(text)
        print(f"文本: {text}")
        print(f"预测: {sentiment}")
        print(f"置信度: {confidence:.4f}")
        
        # 显示所有类别的概率
        print("概率分布:")
        for i, (label, prob) in enumerate(zip(inference.config.label_names, probabilities)):
            print(f"  {label}: {prob:.4f}")
        print("-" * 30)
    
    # 批量预测
    print("\n批量预测结果:")
    results = inference.predict_batch(test_texts)
    
    for result in results:
        text = result['text']
        prediction = result['prediction']
        probabilities = result['probabilities']
        
        label_name = inference.config.label_names[prediction]
        confidence = max(probabilities)
        
        print(f"{text:<30} -> {label_name} ({confidence:.3f})")

if __name__ == "__main__":
    import os
    main() 