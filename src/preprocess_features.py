"""
特征标准化脚本

对已提取的DeBERTa特征进行StandardScaler标准化
解决SVM和MLP的收敛问题

使用方法:
    python preprocess_features.py
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def standardize_features():
    """对特征进行标准化，直接替换原文件"""
    
    logger.info("="*80)
    logger.info("开始特征标准化")
    logger.info("="*80)
    
    # 文件路径
    train_feature_path = 'data/processed/train_features.npy'
    test_feature_path = 'data/processed/test_features.npy'
    
    # 备份路径
    train_backup_path = 'data/processed/train_features_raw.npy'
    test_backup_path = 'data/processed/test_features_raw.npy'
    scaler_path = 'data/processed/scaler.pkl'
    
    # 检查文件是否存在
    if not os.path.exists(train_feature_path):
        logger.error(f"特征文件不存在: {train_feature_path}")
        logger.error("请先运行: python main.py --mode extract")
        return
    
    # 备份原始特征（如果还没备份）
    if not os.path.exists(train_backup_path):
        logger.info("备份原始特征...")
        import shutil
        shutil.copy2(train_feature_path, train_backup_path)
        shutil.copy2(test_feature_path, test_backup_path)
        logger.info(f"✓ 原始特征已备份到: {train_backup_path}, {test_backup_path}")
    else:
        logger.info("检测到已有备份，跳过备份步骤")
    
    # 加载原始特征
    logger.info("\n加载原始特征...")
    X_train = np.load(train_feature_path)
    X_test = np.load(test_feature_path)
    
    logger.info(f"训练集特征形状: {X_train.shape}")
    logger.info(f"测试集特征形状: {X_test.shape}")
    logger.info(f"原始统计: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    
    # 创建StandardScaler
    logger.info("\n使用StandardScaler进行标准化...")
    scaler = StandardScaler()
    
    # 在训练集上fit
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 在测试集上transform
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"标准化后统计: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    
    # 直接替换原文件
    logger.info("\n替换原始特征文件...")
    np.save(train_feature_path, X_train_scaled)
    np.save(test_feature_path, X_test_scaled)
    
    # 保存scaler（用于后续可能的新数据）
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"✓ 训练集已替换: {train_feature_path}")
    logger.info(f"✓ 测试集已替换: {test_feature_path}")
    logger.info(f"✓ Scaler已保存: {scaler_path}")
    
    logger.info("\n" + "="*80)
    logger.info("特征标准化完成!")
    logger.info("="*80)
    logger.info("\n现在可以直接运行实验（无需额外参数）:")
    logger.info("  python main.py --model svm --algo random")
    logger.info("  python main.py --model mlp --algo random")
    logger.info("\n如需恢复原始特征，可以从备份文件恢复：")
    logger.info(f"  {train_backup_path} -> {train_feature_path}")
    logger.info(f"  {test_backup_path} -> {test_feature_path}")


if __name__ == '__main__':
    standardize_features()

