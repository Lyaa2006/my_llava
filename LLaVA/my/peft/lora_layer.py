import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    标准的LoRA层：W_new = W_old + B @ A
    可以动态控制是否使用LoRA
    """
    def __init__(self, original_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.use_lora = True  # 控制是否使用LoRA
        
        # 获取原始层的维度
        if isinstance(original_layer, nn.Linear):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
        else:
            raise ValueError(f"只支持Linear层，但得到了{type(original_layer)}")
        
        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA参数
        self.lora_A = nn.Parameter(
            torch.zeros(self.r, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.r)
        )
        self.dropout = nn.Dropout(dropout)
        
        # 初始化A (Kaiming初始化)，B保持为零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B已经是零
        
    def forward(self, x):
        # 原始前向
        original_output = self.original_layer(x)
        
        if self.use_lora:
            # LoRA分支: x @ A^T @ B^T
            # 注意维度：x: [batch, seq_len, in_features]
            # lora_A: [r, in_features] -> A^T: [in_features, r]
            # lora_B: [out_features, r] -> B^T: [r, out_features]
            lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            return original_output + self.scaling * lora_output
        else:
            return original_output
    
    def merge_lora(self):
        """
        将LoRA参数合并回原始层
        训练完一个task后调用
        """
        if isinstance(self.original_layer, nn.Linear):
            # W_new = W_old + B @ A * scaling
            merged_weight = self.original_layer.weight.data + \
                           self.scaling * (self.lora_B @ self.lora_A)
            self.original_layer.weight.data = merged_weight
            
            # 清零LoRA参数，为下一个任务准备
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
            print(f"LoRA merged into original layer")