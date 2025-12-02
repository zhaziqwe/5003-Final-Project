import numpy as np
import logging
from tqdm import tqdm
from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)


class RandomSearch(BaseHPO):
    
    def __init__(self, n_trials=50, cv_folds=5, random_state=42):
        """
        初始化Random Search
        
        Args:
            n_trials: 搜索次数（默认50）
            cv_folds: 交叉验证折数（默认5）
            random_state: 随机种子（默认42）
        """
        super().__init__(n_trials, cv_folds, random_state)
        np.random.seed(random_state)
        
        # 定义Random Search的搜索空间
        # 格式: (类型, 最小值, 最大值)
        self.search_space = {
            'num_leaves': ('int', 20, 150),
            'max_depth': ('int', 3, 12),
            'learning_rate': ('float_log', 0.01, 0.3),
            'n_estimators': ('int', 100, 1000),
            'min_child_samples': ('int', 10, 100),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'reg_alpha': ('float', 0.0, 10.0),
            'reg_lambda': ('float', 0.0, 10.0),
            'min_split_gain': ('float', 0.0, 1.0),
        }
        
        logger.info(f"搜索空间包含 {len(self.search_space)} 个超参数")
    
    def suggest_params(self):

        params = {}
        
        # 遍历搜索空间，随机采样
        for param_name, param_spec in self.search_space.items():
            param_type, low, high = param_spec
            
            if param_type == 'int':
                # 整数类型
                params[param_name] = np.random.randint(low, high + 1)
            elif param_type == 'float':
                # 浮点类型（线性）
                params[param_name] = np.random.uniform(low, high)
            elif param_type == 'float_log':
                # 浮点类型（对数尺度）
                log_low = np.log(low)
                log_high = np.log(high)
                params[param_name] = np.exp(np.random.uniform(log_low, log_high))
        
        # 添加固定参数
        params.update(self.fixed_params)
        
        return params
    
    def optimize(self, objective_function, verbose=True):

        logger.info(f"开始Random Search，共 {self.n_trials} 次试验")
        logger.info(f"交叉验证: {self.cv_folds} 折")
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
                        logger.info(f"\n发现更好的参数! Trial {trial_idx}")
                        logger.info(f"   得分: {score:.6f}")
                        logger.info(f"   主要参数: lr={params.get('learning_rate', 0):.4f}, "
                                  f"n_estimators={params.get('n_estimators', 0)}, "
                                  f"num_leaves={params.get('num_leaves', 0)}")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} 失败: {str(e)}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("Random Search完成!")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info(f"最佳参数:")
        for key, value in best_params.items():
            if key not in self.fixed_params:  # 只显示可调参数
                logger.info(f"  {key}: {value}")
        logger.info("="*60)
        
        return best_params, best_score
    
    def plot_optimization_history(self, save_path=None):

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
            plt.title('Random Search Optimization History')
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
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")
