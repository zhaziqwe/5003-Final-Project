from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel


class MLPModel(BaseModel):
    """MLP (多层感知机) 模型"""
    
    def __init__(self, params=None, n_folds=5, random_state=42):
        super().__init__(params, n_folds, random_state)
        self.scalers = []  # 每个fold一个scaler
    
    def _create_model(self, params):
        """创建MLP模型"""
        mlp_params = {
            'hidden_layer_sizes': params.get('hidden_layer_sizes', (256, 128)),
            'activation': params.get('activation', 'relu'),
            'solver': 'adam',  # 固定使用adam优化器
            'alpha': params.get('alpha', 0.0001),
            'batch_size': params.get('batch_size', 128),
            'learning_rate': 'adaptive',  # 自适应学习率
            'learning_rate_init': params.get('learning_rate_init', 0.001),
            'max_iter': params.get('max_iter', 500),
            'early_stopping': params.get('early_stopping', True),
            'validation_fraction': 0.1,  # 用于early stopping的验证集比例
            'n_iter_no_change': 10,  # early stopping的patience
            'random_state': self.random_state,
            'verbose': False
        }
        
        return MLPClassifier(**mlp_params)
    
    def _fit(self, model, X_train, y_train, X_val=None, y_val=None):
        """训练MLP模型"""
        # MLP需要标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers.append(scaler)
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        return model
    
    def _predict_proba(self, model, X):
        """预测概率"""
        # 注意：需要使用对应的scaler
        # 这里假设self.scalers已经在train_with_cv中正确设置
        return model.predict_proba(X)
    
    def train_with_cv(self, X, y, verbose=True):
        """重写train_with_cv以正确处理scaler"""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import log_loss
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("="*60)
        logger.info(f"开始交叉验证训练 - {self.__class__.__name__}")
        logger.info(f"训练样本数: {len(X)}, 特征维度: {X.shape[1]}")
        logger.info("="*60)
        
        # 初始化
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        oof_predictions = np.zeros((len(X), 7))
        self.models = []
        self.scalers = []  # 重置scalers
        
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
            
            # 创建scaler并标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers.append(scaler)
            
            # 创建并训练模型
            model = self._create_model(self.params)
            model.fit(X_train_scaled, y_train)
            self.models.append(model)
            
            # 预测验证集
            val_pred = model.predict_proba(X_val_scaled)
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
        
        self.oof_predictions = oof_predictions
        
        return oof_score
    
    def predict(self, X):
        """使用所有fold的模型进行预测并平均"""
        import numpy as np
        
        if not self.models:
            raise ValueError("模型未训练，请先调用train_with_cv()")
        
        predictions = np.zeros((len(X), 7))
        
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred = model.predict_proba(X_scaled)
            predictions += pred
        
        # 平均所有fold的预测
        predictions /= len(self.models)
        
        return predictions

