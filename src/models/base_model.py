import numpy as np
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    模型的抽象基类
    
    所有模型需要实现:
        - _create_model(): 创建模型实例
        - _fit(): 训练模型
        - _predict_proba(): 预测概率
    """
    
    def __init__(self, params=None, n_folds=5, random_state=42):
        """
        初始化模型
        
        Args:
            params: 模型参数字典
            n_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.params = params or {}
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = []  # 存储每个fold的模型
        self.oof_predictions = None
        
        logger.info(f"初始化 {self.__class__.__name__}，{n_folds}折交叉验证")
    
    @abstractmethod
    def _create_model(self, params):
        """
        创建模型实例
        
        Args:
            params: 模型参数
            
        Returns:
            模型实例
        """
        pass
    
    @abstractmethod
    def _fit(self, model, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型
        
        Args:
            model: 模型实例
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            
        Returns:
            训练好的模型
        """
        pass
    
    @abstractmethod
    def _predict_proba(self, model, X):
        """
        预测概率
        
        Args:
            model: 训练好的模型
            X: 特征
            
        Returns:
            预测概率，形状 (n_samples, n_classes)
        """
        pass
    
    def train_with_cv(self, X, y, verbose=True):
        """
        使用交叉验证训练模型
        
        Args:
            X: 特征数组，形状 (n_samples, n_features)
            y: 标签数组，形状 (n_samples,)
            verbose: 是否显示详细信息
            
        Returns:
            float: 交叉验证OOF得分(logloss)
        """
        logger.info("="*60)
        logger.info(f"开始交叉验证训练 - {self.__class__.__name__}")
        logger.info(f"训练样本数: {len(X)}, 特征维度: {X.shape[1]}")
        logger.info("="*60)
        
        # 初始化
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        oof_predictions = np.zeros((len(X), 7))  # Out-of-fold预测
        self.models = []
        
        # 交叉验证
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")
                logger.info(f"{'='*60}")
            
            # 划分数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if verbose:
                logger.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
            
            # 创建并训练模型
            model = self._create_model(self.params)
            model = self._fit(model, X_train, y_train, X_val, y_val)
            
            # 保存模型
            self.models.append(model)
            
            # 预测验证集
            val_pred = self._predict_proba(model, X_val)
            oof_predictions[val_idx] = val_pred
            
            # 计算验证集得分
            fold_score = log_loss(y_val, val_pred)
            cv_scores.append(fold_score)
            
            if verbose:
                logger.info(f"Fold {fold_idx + 1} 验证得分: {fold_score:.6f}")
        
        # 计算OOF得分
        oof_score = log_loss(y, oof_predictions)
        
        logger.info("\n" + "="*60)
        logger.info("交叉验证完成!")
        logger.info(f"各折得分: {[f'{s:.6f}' for s in cv_scores]}")
        logger.info(f"平均得分: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
        logger.info(f"OOF得分: {oof_score:.6f}")
        logger.info("="*60)
        
        # 保存OOF预测
        self.oof_predictions = oof_predictions
        
        return oof_score
    
    def predict(self, X):
        """
        使用所有fold的模型进行预测并平均
        
        Args:
            X: 特征数组
            
        Returns:
            预测概率，形状 (n_samples, n_classes)
        """
        if not self.models:
            raise ValueError("模型未训练，请先调用train_with_cv()")
        
        predictions = np.zeros((len(X), 7))
        
        for model in self.models:
            pred = self._predict_proba(model, X)
            predictions += pred
        
        # 平均所有fold的预测
        predictions /= len(self.models)
        
        return predictions
    
    def save_models(self, save_dir='models'):
        """
        保存所有fold的模型
        
        Args:
            save_dir: 保存目录
        """
        import os
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        for fold_idx, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'{self.__class__.__name__.lower()}_fold_{fold_idx}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"模型已保存到: {save_dir}")
    
    def load_models(self, save_dir='models', n_folds=None):
        """
        加载已保存的模型
        
        Args:
            save_dir: 模型目录
            n_folds: 折数（如果为None则使用self.n_folds）
        """
        import os
        import pickle
        
        if n_folds is None:
            n_folds = self.n_folds
        
        self.models = []
        
        for fold_idx in range(n_folds):
            model_path = os.path.join(save_dir, f'{self.__class__.__name__.lower()}_fold_{fold_idx}.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models.append(model)
        
        logger.info(f"已加载 {len(self.models)} 个模型")

