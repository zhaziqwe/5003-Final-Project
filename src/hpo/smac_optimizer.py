"""
SMAC (Sequential Model-based Algorithm Configuration) using OpenBox
完整实现版本

安装依赖:
pip install openbox

参考文档:
- OpenBox: https://open-box.readthedocs.io/
"""

import logging
from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)


class SMACOptimizer(BaseHPO):
    """
    SMAC/OpenBox - 基于序列模型的超参数优化
    
    使用OpenBox实现，这是一个基于高斯过程的贝叶斯优化方法
    """
    
    def __init__(self, model_name='lightgbm', n_trials=50, cv_folds=5, random_state=42):
        """
        初始化SMAC/OpenBox Optimizer
        
        Args:
            model_name: 模型名称 ('lightgbm', 'svm', 'mlp')
            n_trials: 搜索次数（默认50）
            cv_folds: 交叉验证折数（默认5）
            random_state: 随机种子（默认42）
        """
        super().__init__(model_name, n_trials, cv_folds, random_state)
    
    def optimize(self, objective_function, verbose=True):
        """
        执行SMAC/OpenBox优化
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回logloss
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (最佳参数, 最佳得分)
        """
        try:
            from openbox import Optimizer, sp
        except ImportError:
            raise ImportError("请先安装openbox: pip install openbox")
        
        logger.info(f"开始OpenBox优化，共 {self.n_trials} 次试验")
        logger.info(f"模型: {self.model_name}, 交叉验证: {self.cv_folds} 折")
        logger.info("="*60)
        
        # 定义配置空间
        space = sp.Space()
        
        for param_name, param_spec in self.search_space.items():
            if param_spec['type'] == 'int':
                space.add_variable(sp.Int(
                    param_name, 
                    param_spec['low'], 
                    param_spec['high']
                ))
                
            elif param_spec['type'] == 'float':
                if param_spec.get('log', False):
                    # 对数尺度
                    space.add_variable(sp.Real(
                        param_name, 
                        param_spec['low'], 
                        param_spec['high'],
                        log=True
                    ))
                else:
                    # 线性尺度
                    space.add_variable(sp.Real(
                        param_name, 
                        param_spec['low'], 
                        param_spec['high']
                    ))
                    
            elif param_spec['type'] == 'categorical':
                space.add_variable(sp.Categorical(
                    param_name, 
                    param_spec['choices']
                ))
        
        # OpenBox的objective函数包装器
        def openbox_objective(config):
            # 将config转为字典
            params = config.get_dictionary()
            
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                logger.error(f"评估失败: {e}")
                return float('inf')
        
        # 创建优化器
        opt = Optimizer(
            openbox_objective,
            space,
            max_runs=self.n_trials,
            random_state=self.random_state,
            task_id=f'{self.model_name}_hpo',
            logging_dir='logs',  # 日志目录
            advisor_type='default'  # 使用默认的GP+EI
        )
        
        # 运行优化
        history = opt.run()
        
        # 保存历史
        for i, (config, perf) in enumerate(zip(history.configurations, history.perfs)):
            self.history.append({
                'trial': i,
                'params': config.get_dictionary(),
                'score': perf.objectives[0]  # OpenBox返回的是多目标，我们取第一个
            })
            
            # 显示进度
            if verbose and i % 5 == 0:
                logger.info(f"Trial {i}/{self.n_trials}: Score = {perf.objectives[0]:.6f}")
        
        # 获取最佳结果
        best_config = history.get_best_config()
        best_params = best_config.get_dictionary()
        best_score = history.get_best().objectives[0]
        
        logger.info("\n" + "="*60)
        logger.info("OpenBox优化完成!")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info(f"最佳参数: {best_params}")
        logger.info("="*60)
        
        return best_params, best_score

