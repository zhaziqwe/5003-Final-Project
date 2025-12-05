import numpy as np
import logging
import json
import os
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHPO(ABC):
    """
    HPO算法的抽象基类
    
    子类需要实现:
        - optimize(): 执行优化过程
    """
    
    def __init__(self, model_name='lightgbm', n_trials=50, cv_folds=5, random_state=42):
        """
        初始化HPO算法
        
        Args:
            model_name: 模型名称 ('lightgbm', 'svm', 'mlp')
            n_trials: 搜索次数
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # 加载搜索空间
        self.search_space = self._load_search_space(model_name)
        
        # 存储历史结果
        self.history = []
        
        logger.info(f"初始化 {self.__class__.__name__}")
        logger.info(f"模型: {model_name}, n_trials={n_trials}, cv_folds={cv_folds}")
    
    def _load_search_space(self, model_name):
        """
        从配置文件加载搜索空间
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 搜索空间配置
        """
        # 找到配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'search_spaces.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"搜索空间配置文件不存在: {config_path}")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            all_spaces = json.load(f)
        
        if model_name not in all_spaces:
            raise ValueError(f"模型 '{model_name}' 的搜索空间未定义")
        
        search_space = all_spaces[model_name]
        logger.info(f"已加载模型 '{model_name}' 的搜索空间，包含 {len(search_space)} 个超参数")
        
        return search_space
    
    def suggest_params(self):
        """
        随机建议一组超参数（默认实现：Random Search）
        子类可以重写此方法实现更高级的采样策略
        
        Returns:
            dict: 超参数字典
        """
        import random
        
        params = {}
        
        for param_name, param_spec in self.search_space.items():
            param_type = param_spec['type']
            
            if param_type == 'int':
                low, high = param_spec['low'], param_spec['high']
                params[param_name] = np.random.randint(low, high + 1)
                
            elif param_type == 'float':
                low, high = param_spec['low'], param_spec['high']
                if param_spec.get('log', False):
                    # 对数尺度采样
                    log_low = np.log(low)
                    log_high = np.log(high)
                    params[param_name] = np.exp(np.random.uniform(log_low, log_high))
                else:
                    # 线性尺度采样
                    params[param_name] = np.random.uniform(low, high)
                    
            elif param_type == 'categorical':
                choices = param_spec['choices']
                # 使用Python的random.choice，支持嵌套列表
                params[param_name] = random.choice(choices)
        
        return params
    
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
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        logger.info(f"优化历史已保存到: {save_path}")
    
    def plot_optimization_history(self, save_path=None):
        """
        绘制优化历史（通用版本）
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.history:
                logger.warning("没有历史记录可以绘制")
                return
            
            trials = [h['trial'] for h in self.history]
            scores = [h['score'] for h in self.history]
            
            # 计算累积最佳得分
            best_scores = []
            current_best = float('inf')
            for score in scores:
                current_best = min(current_best, score)
                best_scores.append(current_best)
            
            # 绘图
            plt.figure(figsize=(12, 5))
            
            # 子图1: 所有试验的得分
            plt.subplot(1, 2, 1)
            plt.scatter(trials, scores, alpha=0.6, s=30)
            plt.plot(trials, best_scores, 'r-', linewidth=2, label='Best Score')
            plt.xlabel('Trial')
            plt.ylabel('Score (logloss)')
            plt.title(f'{self.__class__.__name__} - {self.model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: 得分分布
            plt.subplot(1, 2, 2)
            plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(min(scores), color='r', linestyle='--', linewidth=2, label=f'Best: {min(scores):.6f}')
            plt.xlabel('Score (logloss)')
            plt.ylabel('Frequency')
            plt.title('Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"优化历史图已保存到: {save_path}")
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")
