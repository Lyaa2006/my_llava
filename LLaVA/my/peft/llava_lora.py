import torch
import torch.nn as nn
from .lora_layer import LoRALayer

class LLaVAWithLoRA(nn.Module):
    """
    为LLaVA模型添加LoRA支持（针对 SSP-CL 优化）
    支持只给指定层（16-31）的 v_proj 和 mm_projector 添加 LoRA
    """
    def __init__(self, base_model, lora_config=None):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config or {
            'r': 8,
            'alpha': 16,
            'target_layers': list(range(16, 32)),
            'target_modules': ['v_proj'],
            'mm_projector': True,
            'mm_projector_r': 8,
            'mm_projector_alpha': 16,
            'dropout': 0.1
        }
        
        print("\n" + "="*60)
        print("LLaVAWithLoRA 启动：正在构建参数高效微调层...")
        
        # 1. 冻结基础模型全部参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.lora_layers = nn.ModuleDict() # 使用 ModuleDict 确保参数被正确注册
        self._inject_lora()
        
        print(f"✓ 成功注入 {len(self.lora_layers)} 个 LoRA 模块")
        print("="*60 + "\n")

    @property
    def device(self):
        return self.base_model.device

    @property
    def dtype(self):
        return self.base_model.dtype

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _inject_lora(self):
        if not hasattr(self.base_model, 'get_model'):
            return
        llm_model = self.base_model.get_model()

        # --- A. LLM 注入 ---
        if hasattr(llm_model, 'layers'):
            for idx, layer in enumerate(llm_model.layers):
                if idx in self.lora_config['target_layers']:
                    attn = layer.self_attn
                    for target in self.lora_config['target_modules']:
                        if hasattr(attn, target):
                            origin_mod = getattr(attn, target)
                            lora_mod = LoRALayer(origin_mod, r=self.lora_config['r'],
                                               alpha=self.lora_config['alpha'],
                                               dropout=self.lora_config['dropout'])
                            setattr(attn, target, lora_mod)
                            self.lora_layers[f"llm_layer_{idx}_{target}"] = lora_mod

        # --- B. 投影层注入 ---
        if self.lora_config.get('mm_projector', False):
            proj = None
            parent = None
            attr = None
            if hasattr(self.base_model, 'mm_projector'):
                proj, parent, attr = self.base_model.mm_projector, self.base_model, 'mm_projector'
            elif hasattr(llm_model, 'mm_projector'):
                proj, parent, attr = llm_model.mm_projector, llm_model, 'mm_projector'
            
            if isinstance(proj, nn.Sequential):
                for i, m in enumerate(proj):
                    if isinstance(m, nn.Linear):
                        lora_m = LoRALayer(m, r=self.lora_config['mm_projector_r'],
                                         alpha=self.lora_config['mm_projector_alpha'])
                        proj[i] = lora_m
                        self.lora_layers[f"mm_projector_{i}"] = lora_m
            elif isinstance(proj, nn.Linear):
                lora_m = LoRALayer(proj, r=self.lora_config['mm_projector_r'],
                                  alpha=self.lora_config['mm_projector_alpha'])
                setattr(parent, attr, lora_m)
                self.lora_layers["mm_projector"] = lora_m

    # --- 新增：持续学习核心重置逻辑 ---

    def merge_all_lora(self):
        """将当前学习到的知识永久合并到 Base Model"""
        print("正在合并当前任务的 LoRA 权重到主干网络...")
        for name, layer in self.lora_layers.items():
            if hasattr(layer, 'merge_lora'):
                layer.merge_lora()
            # 合并后，确保 LoRA 内部的参数不再接受梯度，直到被 reset
            for p in layer.parameters():
                p.requires_grad = False

    def reset_lora(self):
        """
        重置 LoRA 参数（A/B 矩阵）。
        这在任务切换时调用，使得下一个任务从全新的增量开始，避免干扰。
        """
        print("正在重置 LoRA 矩阵（A 高斯分布，B 全零）...")
        for layer in self.lora_layers.values():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters() 
            # 确保重置后的参数重新开启梯度
            for p in layer.parameters():
                p.requires_grad = True

    # --- 原有转发逻辑保持不变 ---

    def set_lora_active(self, active=True):
        for layer in self.lora_layers.values():
            layer.use_lora = active

    def forward(self, **kwargs):
        use_lora = kwargs.pop('use_lora', True)
        self.set_lora_active(use_lora)
        return self.base_model(**kwargs)

    @torch.no_grad()
    def generate(self, **kwargs):
        self.set_lora_active(False)
        return self.base_model.generate(**kwargs)

    def get_model(self):
        return self.base_model.get_model()

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)