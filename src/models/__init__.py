from .lgb_model import LightGBMModel
from .svm_model import SVMModel
from .mlp_model import MLPModel

# 模型注册表 - 队友添加新模型只需在这里注册
AVAILABLE_MODELS = {
    'lightgbm': LightGBMModel,
    'svm': SVMModel,
    'mlp': MLPModel,
}


def get_model(model_name):
    """
    获取模型类
    
    Args:
        model_name: 模型名称（如 'lightgbm', 'svm', 'mlp'）
        
    Returns:
        模型类
        
    Raises:
        ValueError: 如果模型名称不存在
    """
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"未知的模型: '{model_name}'\n"
            f"可用的模型: {available}"
        )
    return AVAILABLE_MODELS[model_name]


def list_models():
    """列出所有可用的模型"""
    return list(AVAILABLE_MODELS.keys())


__all__ = ['LightGBMModel', 'SVMModel', 'MLPModel', 'get_model', 'list_models', 'AVAILABLE_MODELS']

