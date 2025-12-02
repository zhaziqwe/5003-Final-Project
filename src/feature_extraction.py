import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeBERTaFeatureExtractor:
    """
    DeBERTa特征提取器
    用于从文本中提取固定长度的向量特征
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-base', max_length=512, batch_size=8):
        """
        初始化特征提取器
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"加载模型: {model_name}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        logger.info("模型加载完成")
    
    def extract_features(self, texts, show_progress=True):
        """
        从文本列表中提取特征
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            numpy数组，形状为 (n_samples, hidden_size)
        """
        features = []
        
        # 使用tqdm显示进度
        iterator = tqdm(range(0, len(texts), self.batch_size), 
                       desc="提取特征", 
                       disable=not show_progress)
        
        with torch.no_grad():  # 不计算梯度，节省内存
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                
                # 确保所有文本都是字符串类型（处理NaN等情况）
                batch_texts = [str(text) if text is not None else "" for text in batch_texts]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 使用[CLS] token的输出作为句子表示
                # last_hidden_state的形状: (batch_size, sequence_length, hidden_size)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                features.append(cls_embeddings)
        
        # 合并所有批次的特征
        features = np.vstack(features)
        logger.info(f"特征提取完成，形状: {features.shape}")
        
        return features
    
    def process_dataset(self, df, text_column='combined_text', save_path=None):
        """
        处理整个数据集
        
        Args:
            df: pandas DataFrame
            text_column: 文本列名
            save_path: 保存路径（可选）
            
        Returns:
            特征数组
        """
        logger.info(f"开始处理数据集，共 {len(df)} 条样本")
        
        # 确保文本列没有空值
        df[text_column] = df[text_column].fillna("").astype(str)
        
        # 提取特征
        features = self.extract_features(df[text_column].tolist())
        
        # 保存特征
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features)
            logger.info(f"特征已保存到: {save_path}")
        
        return features


def prepare_text_data(df, question_col='Question', response_col='Response'):
    """
    准备文本数据，将问题和回答组合
    
    Args:
        df: pandas DataFrame
        question_col: 问题列名
        response_col: 回答列名
        
    Returns:
        包含组合文本的DataFrame
    """
    df = df.copy()
    
    # 填充空值，确保所有文本都是字符串
    df[question_col] = df[question_col].fillna("").astype(str)
    df[response_col] = df[response_col].fillna("").astype(str)
    
    # 组合问题和回答，使用[SEP]分隔
    df['combined_text'] = df[question_col] + " [SEP] " + df[response_col]
    
    logger.info(f"文本数据准备完成，共 {len(df)} 条")
    return df


def extract_and_save_features(
    train_path='data/raw/train.csv',
    test_path='data/raw/test.csv',
    train_feature_path='data/processed/train_features.npy',
    test_feature_path='data/processed/test_features.npy',
    train_labels_path='data/processed/train_labels.npy',
    test_ids_path='data/processed/test_ids.npy',
    model_name='microsoft/deberta-v3-base',
    max_length=512,
    batch_size=8
):
    """
    提取并保存训练集和测试集的特征
    
    这是主函数，一次性处理所有数据
    
    Args:
        train_path: 训练集路径
        test_path: 测试集路径
        train_feature_path: 训练特征保存路径
        test_feature_path: 测试特征保存路径
        train_labels_path: 训练标签保存路径
        test_ids_path: 测试ID保存路径
        model_name: 模型名称
        max_length: 最大序列长度
        batch_size: 批处理大小
    """
    # 读取数据
    logger.info("="*50)
    logger.info("开始特征提取流程")
    logger.info("="*50)
    
    logger.info(f"读取训练数据: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"训练集样本数: {len(train_df)}")
    
    logger.info(f"读取测试数据: {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"测试集样本数: {len(test_df)}")
    
    # 准备文本数据
    train_df = prepare_text_data(train_df)
    test_df = prepare_text_data(test_df)
    
    # 初始化特征提取器
    extractor = DeBERTaFeatureExtractor(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # 提取训练集特征
    logger.info("\n处理训练集...")
    train_features = extractor.process_dataset(
        train_df, 
        text_column='combined_text',
        save_path=train_feature_path
    )
    
    # 保存训练标签
    train_labels = train_df['target'].values
    os.makedirs(os.path.dirname(train_labels_path), exist_ok=True)
    np.save(train_labels_path, train_labels)
    logger.info(f"训练标签已保存到: {train_labels_path}")
    
    # 提取测试集特征
    logger.info("\n处理测试集...")
    test_features = extractor.process_dataset(
        test_df,
        text_column='combined_text',
        save_path=test_feature_path
    )
    
    # 保存测试ID
    test_ids = test_df['id'].values
    os.makedirs(os.path.dirname(test_ids_path), exist_ok=True)
    np.save(test_ids_path, test_ids)
    logger.info(f"测试ID已保存到: {test_ids_path}")
    
    logger.info("\n"+"="*50)
    logger.info("特征提取完成！")
    logger.info(f"训练特征形状: {train_features.shape}")
    logger.info(f"测试特征形状: {test_features.shape}")
    logger.info("="*50)


if __name__ == '__main__':
    # 直接运行此脚本即可提取特征
    extract_and_save_features()

