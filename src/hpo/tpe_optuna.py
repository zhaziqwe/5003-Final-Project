"""
TPE (Tree-structured Parzen Estimator) using Optuna
完整实现版本

安装依赖:
pip install optuna

参考文档:
https://optuna.readthedocs.io/
"""

import logging
from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)


class TPEOptuna(BaseHPO):
    """
    TPE (Tree-structured Parzen Estimator) - 使用Optuna实现
    
    TPE是一种基于贝叶斯优化的方法，通过建立概率模型来智能采样
    """
    
    def __init__(self, model_name='lightgbm', n_trials=50, cv_folds=5, random_state=42):
        """
        初始化TPE Optimizer
        
        Args:
            model_name: 模型名称 ('lightgbm', 'svm', 'mlp')
            n_trials: 搜索次数（默认50）
            cv_folds: 交叉验证折数（默认5）
            random_state: 随机种子（默认42）
        """
        super().__init__(model_name, n_trials, cv_folds, random_state)
        
        # Optuna相关
        self.study = None  # Optuna Study对象
    
    def optimize(self, objective_function, verbose=True):
        """
        执行TPE优化
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回logloss
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (最佳参数, 最佳得分)
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError("请先安装optuna: pip install optuna")
        
        logger.info(f"开始TPE优化，共 {self.n_trials} 次试验")
        logger.info(f"模型: {self.model_name}, 交叉验证: {self.cv_folds} 折")
        logger.info("="*60)
        
        # 创建Optuna Study
        sampler = TPESampler(seed=self.random_state)
        
        # 关闭optuna的日志输出
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        # 定义objective函数包装器
        def optuna_objective(trial):
            # 根据self.search_space定义参数
            params = {}
            
            for param_name, param_spec in self.search_space.items():
                if param_spec['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_spec['low'], 
                        param_spec['high']
                    )
                    
                elif param_spec['type'] == 'float':
                    if param_spec.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_spec['low'], 
                            param_spec['high'],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_spec['low'], 
                            param_spec['high']
                        )
                        
                elif param_spec['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_spec['choices']
                    )
            
            # 调用objective_function
            try:
                score = objective_function(params)
                
                # 显示进度
                if verbose and trial.number % 5 == 0:
                    logger.info(f"Trial {trial.number}/{self.n_trials}: Score = {score:.6f}")
                
                return score
            except Exception as e:
                logger.error(f"Trial {trial.number} 失败: {e}")
                return float('inf')  # 返回一个很大的值
        
        # 运行优化
        self.study.optimize(
            optuna_objective, 
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )
        
        # 保存历史
        for trial in self.study.trials:
            if trial.value is not None:  # 只保存成功的试验
                self.history.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'score': trial.value
                })
        
        # 获取最佳结果
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        logger.info("\n" + "="*60)
        logger.info("TPE优化完成!")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info(f"最佳参数: {best_params}")
        logger.info("="*60)
        
        return best_params, best_score
    
    def plot_optimization_history(self, save_path=None):
        """
        绘制优化历史
        使用父类的默认可视化方法
        """
        # 使用父类的方法
        super().plot_optimization_history(save_path)
        
        # 如果需要使用Optuna的可视化（需要安装plotly和kaleido）
        # 可以取消下面的注释
        # try:
        #     import optuna.visualization as vis
        #     if self.study is not None:
        #         fig = vis.plot_optimization_history(self.study)
        #         fig.write_image(save_path.replace('.png', '_optuna.png'))
        # except ImportError:
        #     logger.debug("plotly未安装，使用默认可视化")
        # except Exception as e:
        #     logger.debug(f"Optuna可视化失败: {e}")

