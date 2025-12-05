import argparse
import numpy as np
import logging
import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction import extract_and_save_features
from models import get_model, list_models
from hpo import get_hpo_algorithm, list_algorithms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_feature_extraction(args):
    """运行特征提取"""
    logger.info("="*80)
    logger.info("DeBERTa特征提取")
    logger.info("="*80)
    
    extract_and_save_features(
        train_path=args.train_path,
        test_path=args.test_path,
        train_feature_path=args.train_feature_path,
        test_feature_path=args.test_feature_path,
        train_labels_path=args.train_labels_path,
        test_ids_path=args.test_ids_path,
        model_name=args.deberta_model,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    logger.info("\n特征提取完成！")


def objective_function_factory(model_class, X, y, n_folds=5):
    """
    创建目标函数（用于HPO）
    
    Args:
        model_class: 模型类
        X: 特征数组
        y: 标签数组
        n_folds: 交叉验证折数
        
    Returns:
        objective_function: 接受参数字典，返回logloss的函数
    """
    def objective(params):
        """
        目标函数
        
        Args:
            params: 超参数字典
            
        Returns:
            float: 交叉验证OOF logloss
        """
        # 创建模型
        model = model_class(params=params, n_folds=n_folds)
        
        # 训练并获取CV得分
        cv_score = model.train_with_cv(X, y, verbose=False)
        
        return cv_score
    
    return objective


def run_hpo(args):
    """
    运行超参数优化（核心函数）
    支持多模型、多HPO算法
    """
    logger.info("="*80)
    logger.info(f"HPO超参数优化")
    logger.info(f"模型: {args.model} | HPO算法: {args.hpo_algorithm}")
    logger.info("="*80)
    
    # 检查特征文件是否存在
    if not os.path.exists(args.train_feature_path):
        logger.error(f"特征文件不存在: {args.train_feature_path}")
        logger.error("请先运行: python main.py --mode extract")
        return None, None
    
    # 加载训练数据
    logger.info("加载训练特征...")
    X_train = np.load(args.train_feature_path)
    y_train = np.load(args.train_labels_path)
    X_test = np.load(args.test_feature_path)
    test_ids = np.load(args.test_ids_path)
    logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 获取模型类
    ModelClass = get_model(args.model)
    
    # 获取HPO算法类
    HPOClass = get_hpo_algorithm(args.hpo_algorithm)
    
    # 初始化HPO算法
    optimizer = HPOClass(
        model_name=args.model,
        n_trials=args.n_trials,
        cv_folds=5,
        random_state=42
    )
    
    # 创建目标函数
    objective_fn = objective_function_factory(
        ModelClass, 
        X_train, 
        y_train,
        n_folds=5
    )
    
    # 执行优化
    best_params, best_score = optimizer.optimize(
        objective_function=objective_fn,
        verbose=True
    )
    
    # 保存结果
    os.makedirs(args.model_dir, exist_ok=True)
    history_path = os.path.join(
        args.model_dir, 
        f'{args.model}_{args.hpo_algorithm}_history.json'
    )
    optimizer.save_history(history_path)
    logger.info(f"优化历史已保存: {history_path}")
    
    # 绘制优化历史
    try:
        plot_path = os.path.join(
            args.model_dir, 
            f'{args.model}_{args.hpo_algorithm}_history.png'
        )
        optimizer.plot_optimization_history(plot_path)
        logger.info(f"可视化图已保存: {plot_path}")
    except Exception as e:
        logger.debug(f"无法绘制优化历史: {e}")
    
    # 使用最佳参数训练最终模型并生成提交文件
    logger.info("\n" + "="*80)
    logger.info("使用最佳参数训练最终模型")
    logger.info("="*80)
    
    import pandas as pd
    
    trainer = ModelClass(params=best_params, n_folds=5)
    final_cv_score = trainer.train_with_cv(X_train, y_train)
    
    # 保存模型
    trainer.save_models(args.model_dir)
    
    # 预测测试集
    logger.info("\n预测测试集...")
    test_predictions = trainer.predict(X_test)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'target_0': test_predictions[:, 0],
        'target_1': test_predictions[:, 1],
        'target_2': test_predictions[:, 2],
        'target_3': test_predictions[:, 3],
        'target_4': test_predictions[:, 4],
        'target_5': test_predictions[:, 5],
        'target_6': test_predictions[:, 6]
    })
    
    # 修改输出文件名以包含模型和算法信息
    output_dir = os.path.dirname(args.output_path)
    output_filename = f'{args.model}_{args.hpo_algorithm}_submission.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    logger.info(f"\n提交文件已保存: {output_path}")
    logger.info(f"最终CV得分: {final_cv_score:.6f}")
    
    return best_params, best_score


def main():
    """
    主函数 - 超级简单版
    
    使用示例:
        python main.py --mode extract                              # 提取特征
        python main.py --model lightgbm --algo random              # LightGBM + Random Search
        python main.py --model svm --algo grid                     # SVM + Grid Search
        python main.py --model mlp --algo tpe                      # MLP + TPE
    """
    parser = argparse.ArgumentParser(
        description='DSAA5003 Final Project - 多模型多HPO方法对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode extract                       # 提取DeBERTa特征
  python main.py --model lightgbm --algo random       # LightGBM + Random Search
  python main.py --model svm --algo grid              # SVM + Grid Search  
  python main.py --model mlp --algo tpe               # MLP + TPE
  python main.py --model lightgbm --algo smac         # LightGBM + SMAC
        """
    )
    
    # ===== 核心参数 =====
    parser.add_argument('--mode', type=str, default='hpo',
                       choices=['extract', 'hpo'],
                       help='运行模式: extract(特征提取) 或 hpo(超参数优化)')
    
    parser.add_argument('--model', type=str, default='lightgbm',
                       choices=list_models(),
                       help=f'模型名称，可用: {list_models()}')
    
    parser.add_argument('--algo', type=str, default='random',
                       choices=list_algorithms(),
                       help=f'HPO算法名称，可用: {list_algorithms()}')
    
    parser.add_argument('--n_trials', type=int, default=50,
                       help='HPO搜索次数（默认50）')
    
    args = parser.parse_args()
    
    # ===== 固定配置 =====
    args.hpo_algorithm = args.algo  # 统一使用algo
    args.train_path = 'data/raw/train.csv'
    args.test_path = 'data/raw/test.csv'
    args.train_feature_path = 'data/processed/train_features.npy'
    args.test_feature_path = 'data/processed/test_features.npy'
    args.train_labels_path = 'data/processed/train_labels.npy'
    args.test_ids_path = 'data/processed/test_ids.npy'
    args.output_path = 'outputs/submission.csv'
    args.model_dir = 'models'
    args.deberta_model = 'microsoft/deberta-v3-base'
    args.max_length = 512
    args.batch_size = 8
    
    # 打印配置
    logger.info("="*80)
    logger.info("DSAA5003 Final Project - 多模型多HPO对比")
    logger.info("="*80)
    logger.info(f"运行模式: {args.mode}")
    if args.mode == 'hpo':
        logger.info(f"模型: {args.model}")
        logger.info(f"HPO算法: {args.hpo_algorithm}")
        logger.info(f"搜索次数: {args.n_trials}")
    logger.info("="*80 + "\n")
    
    try:
        if args.mode == 'extract':
            # 特征提取
            run_feature_extraction(args)
            
        elif args.mode == 'hpo':
            # HPO优化
            run_hpo(args)
        
        logger.info("\n" + "="*80)
        logger.info("任务完成!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n发生错误: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
