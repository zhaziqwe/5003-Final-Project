import numpy as np
import logging
from tqdm import tqdm
from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)


class RandomSearch(BaseHPO):
    """
    Random Search - 随机搜索超参数优化
    
    最简单的HPO方法，随机采样参数空间
    """
    
    def __init__(self, model_name='lightgbm', n_trials=50, cv_folds=5, random_state=42):
        """
        初始化Random Search
        
        Args:
            model_name: 模型名称 ('lightgbm', 'svm', 'mlp')
            n_trials: 搜索次数（默认50）
            cv_folds: 交叉验证折数（默认5）
            random_state: 随机种子（默认42）
        """
        super().__init__(model_name, n_trials, cv_folds, random_state)
        np.random.seed(random_state)
        
        logger.info(f"搜索空间包含 {len(self.search_space)} 个超参数")
    
    def optimize(self, objective_function, verbose=True):
        """
        执行Random Search优化
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回logloss
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (最佳参数, 最佳得分)
        """
        logger.info(f"开始Random Search，共 {self.n_trials} 次试验")
        logger.info(f"模型: {self.model_name}, 交叉验证: {self.cv_folds} 折")
        logger.info("="*60)
        
        best_score = float('inf')
        best_params = None
        
        # 使用tqdm显示进度
        iterator = tqdm(range(self.n_trials), desc="Random Search进度") if verbose else range(self.n_trials)
        
        for trial_idx in iterator:
            # 随机采样一组超参数
            params = self.suggest_params()
            
            try:
                # 评估超参数
                score = objective_function(params)
                
                # 记录历史
                self.history.append({
                    'trial': trial_idx,
                    'params': params,
                    'score': score
                })
                
                # 更新最佳结果
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    
                    if verbose:
                        logger.info(f"\n✓ 发现更好的参数! Trial {trial_idx}, Score: {score:.6f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} 失败: {str(e)}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("Random Search完成!")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info(f"最佳参数: {best_params}")
        logger.info("="*60)
        
        return best_params, best_score
