# __init__.py
"""
PEFT (Parameter-Efficient Fine-Tuning) 模块，用于LLaVA的持续学习
包含LoRA实现、LLaVA模型包装和持续学习训练器
"""

from .lora_layer import LoRALayer
from .llava_lora import LLaVAWithLoRA
from .continual_trainer import ContinualLoRATrainer

__all__ = [
    'LoRALayer',
    'LLaVAWithLoRA', 
    'ContinualLoRATrainer',
]

__version__ = '0.1.0'
__author__ = 'Your Name'
__description__ = 'LoRA implementation for continual learning with LLaVA'