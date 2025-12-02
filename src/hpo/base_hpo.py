import numpy as np
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHPO(ABC):
    """
    HPO算法的抽象基类
    
    子类需要实现:
        - suggest_params(): 建议下一组超参数
        - optimize(): 执行优化过程
    """
    
    def __init__(self, n_trials=50, cv_folds=5, random_state=42):
        """
        初始化HPO算法
        
        Args:
            n_trials: 搜索次数
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # LightGBM固定参数（所有算法共享）
        self.fixed_params = {
            'objective': 'multiclass',
            'num_class': 7,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # 存储历史结果
        self.history = []
        
        logger.info(f"初始化 {self.__class__.__name__}")
        logger.info(f"n_trials={n_trials}, cv_folds={cv_folds}, random_state={random_state}")
    
    @abstractmethod
    def suggest_params(self):
        """
        建议下一组超参数
        
        Returns:
            dict: 超参数字典
        """
        pass
    
    @abstractmethod
    def optimize(self, objective_function, verbose=True):
        """
        执行优化过程
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回评分
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (最佳参数, 最佳得分)
        """
        pass
    
    def get_best_params(self):
        """
        获取历史最佳参数
        
        Returns:
            dict: 最佳参数字典
        """
        if not self.history:
            logger.warning("没有历史记录")
            return None
        
        # 找到得分最低的（logloss越小越好）
        best_idx = np.argmin([h['score'] for h in self.history])
        return self.history[best_idx]['params']
    
    def get_best_score(self):
        """
        获取历史最佳得分
        
        Returns:
            float: 最佳得分
        """
        if not self.history:
            logger.warning("没有历史记录")
            return None
        
        return min([h['score'] for h in self.history])
    
    def save_history(self, save_path):
        """
        保存优化历史
        
        Args:
            save_path: 保存路径
        """
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        logger.info(f"优化历史已保存到: {save_path}")
