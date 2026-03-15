import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    标准的LoRA层：W_new = W_old + (B @ A) * scaling
    针对持续学习优化：支持显式权重合并与参数重置
    """
    def __init__(self, original_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        
        if not isinstance(original_layer, nn.Linear):
            raise ValueError(f"LoRALayer 目前仅支持 nn.Linear 层，收到: {type(original_layer)}")

        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.use_lora = True  # 全局开关

        # 1. 冻结原始层参数
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # 2. 定义 LoRA 参数
        # 我们使用 original_layer 的 dtype 和 device 来初始化，避免后续 forward 中的转换
        factory_kwargs = {'device': original_layer.weight.device, 'dtype': original_layer.weight.dtype}
        
        self.lora_A = nn.Parameter(torch.empty((r, original_layer.in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty((original_layer.out_features, r), **factory_kwargs))
        self.dropout = nn.Dropout(dropout)
        
        # 3. 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置 LoRA 参数：A 使用 Kaiming 初始化，B 清零。
        这确保了任务开始时 LoRA 分支对原始权重的干扰为 0。
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge_lora(self):
        """
        将当前任务学到的 Delta W 合并回 original_layer 的权重中。
        """
        with torch.no_grad():
            # W = W + (B @ A) * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data.add_(delta_w)
            # 合并后通常需要重置 B，防止在 forward 中重复叠加已沉淀的知识
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始权重输出
        result = self.original_layer(x)
        
        if self.use_lora:
            # 这里的计算会自动跟随 x 的精度和设备，前提是初始化时已对齐
            # 避免在 forward 中调用 .to()，那会导致严重的性能瓶颈
            
            # 分支路径: x -> dropout -> A -> B -> scaling
            x_dtype = x.dtype
            # 确保 LoRA 计算在混合精度下也是安全的
            lora_out = (self.dropout(x) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
            
            result += lora_out * self.scaling
            
        return result

    def extra_repr(self) -> str:
        return f"in_features={self.original_layer.in_features}, out_features={self.original_layer.out_features}, r={self.r}, alpha={self.alpha}"