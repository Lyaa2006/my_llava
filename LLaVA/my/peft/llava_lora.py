import torch
import torch.nn as nn
from .lora_layer import LoRALayer

class LLaVAWithLoRA(nn.Module):
    """
    为LLaVA模型添加LoRA支持（包括LLM和投影层）
    优化：只给16-31层的v_proj层添加LoRA
    """
    def __init__(self, base_model, lora_config=None):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config or {
            'r': 8, 
            'alpha': 16, 
            'target_layers': list(range(16, 32)),  # 只选择16-31层
            'target_modules': ['v_proj'],  # 只选择v_proj
            'mm_projector': True,  # 是否给投影层加LoRA
            'mm_projector_r': 8,   # 投影层的LoRA秩
            'mm_projector_alpha': 16,  # 投影层的LoRA alpha
            'dropout': 0.1
        }
        
        print("\n" + "="*60)
        print("LLaVAWithLoRA 初始化开始")
        print("="*60)
        
        # 打印配置信息
        print(f"LoRA配置:")
        print(f"  - 目标层: {self.lora_config['target_layers']}")
        print(f"  - 目标模块: {self.lora_config['target_modules']}")
        print(f"  - LoRA秩: {self.lora_config['r']}")
        print(f"  - 投影层LoRA: {self.lora_config['mm_projector']}")
        
        # 打印base_model类型
        print(f"\nbase_model 类型: {type(self.base_model)}")
        
        # 冻结整个base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("✓ 基础模型已冻结")
            
        # 记录被LoRA替换的层
        self.lora_layers = {}
        self._inject_lora()
        
        print("="*60)
        print("LLaVAWithLoRA 初始化完成")
        print("="*60 + "\n")
    
    def _inject_lora(self):
        """
        在LLM的指定层（16-31层）的v_proj注入LoRA
        """
        print("\n----- 开始注入LoRA -----")
        
        # 获取LLM模型
        if hasattr(self.base_model, 'get_model'):
            llm_model = self.base_model.get_model()
            print(f"✓ 获取到LLM模型，类型: {type(llm_model)}")
        else:
            print("✗ base_model 没有 get_model() 方法")
            return
        
        # ==== 1. 给LLM注入LoRA（只给16-31层的v_proj）====
        print("\n----- 1. 注入LoRA到LLM -----")
        print(f"目标层: {self.lora_config['target_layers']}")
        print(f"目标模块: {self.lora_config['target_modules']}")
        
        if hasattr(llm_model, 'layers'):
            layers = llm_model.layers
            total_layers = len(layers)
            print(f"模型共有 {total_layers} 层transformer层")
            
            replaced_count = 0
            for layer_idx, layer in enumerate(layers):
                # 只处理目标层（16-31层）
                if layer_idx in self.lora_config['target_layers']:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        
                        for target in self.lora_config['target_modules']:
                            if hasattr(attn, target):
                                module = getattr(attn, target)
                                if isinstance(module, nn.Linear):
                                    layer_name = f"llm.layers.{layer_idx}.self_attn.{target}"
                                    print(f"  替换LLM: {layer_name}")
                                    
                                    lora_layer = LoRALayer(
                                        module, 
                                        r=self.lora_config['r'],
                                        alpha=self.lora_config['alpha'],
                                        dropout=self.lora_config['dropout']
                                    )
                                    
                                    setattr(attn, target, lora_layer)
                                    self.lora_layers[layer_name] = lora_layer
                                    replaced_count += 1
            
            print(f"\nLLM注入完成，共替换 {replaced_count} 个层")
            print(f"  - 替换的层索引: {sorted([idx for idx in self.lora_config['target_layers'] if idx < total_layers])}")
        else:
            print("警告：llm_model 没有 'layers' 属性")
        
        # ==== 2. 给投影层注入LoRA（保持不变）====
        print("\n----- 2. 注入LoRA到投影层 -----")
        if self.lora_config.get('mm_projector', False):
            print(f"投影层LoRA配置: r={self.lora_config.get('mm_projector_r', 8)}, alpha={self.lora_config.get('mm_projector_alpha', 16)}")
            
            # 尝试多个可能的位置找到投影层
            mm_projector = None
            found_location = None
            
            # 按优先级检查各个位置
            locations_to_check = [
                ('base_model.mm_projector', self.base_model, 'mm_projector'),
                ('base_model.model.mm_projector', self.base_model.model if hasattr(self.base_model, 'model') else None, 'mm_projector'),
                ('base_model.base_model.mm_projector', self.base_model.base_model if hasattr(self.base_model, 'base_model') else None, 'mm_projector'),
            ]
            
            for loc_name, obj, attr_name in locations_to_check:
                if obj is not None and hasattr(obj, attr_name):
                    mm_projector = getattr(obj, attr_name)
                    found_location = loc_name
                    print(f"✓ 在 {found_location} 找到投影层")
                    break
            
            if mm_projector is not None:
                print(f"投影层类型: {type(mm_projector)}")
                
                # 处理不同类型的投影层
                if isinstance(mm_projector, nn.Sequential):
                    print(f"投影层是 Sequential，包含 {len(mm_projector)} 个子层")
                    
                    # 打印每一层的类型
                    for i, layer in enumerate(mm_projector):
                        print(f"  层{i}: {type(layer)}")
                    
                    # 创建新的层列表
                    new_layers = []
                    replaced_count = 0
                    
                    for proj_idx, proj_layer in enumerate(mm_projector):
                        if isinstance(proj_layer, nn.Linear):
                            layer_name = f"mm_projector.{proj_idx}"
                            print(f"  替换投影层: {layer_name}")
                            
                            lora_layer = LoRALayer(
                                proj_layer,
                                r=self.lora_config.get('mm_projector_r', 8),
                                alpha=self.lora_config.get('mm_projector_alpha', 16),
                                dropout=self.lora_config.get('dropout', 0.1)
                            )
                            new_layers.append(lora_layer)
                            self.lora_layers[layer_name] = lora_layer
                            replaced_count += 1
                        else:
                            new_layers.append(proj_layer)
                    
                    # 创建新的Sequential
                    new_mm_projector = nn.Sequential(*new_layers)
                    
                    # 根据找到的位置设置回去
                    if found_location == 'base_model.mm_projector':
                        self.base_model.mm_projector = new_mm_projector
                        print(f"✓ 已更新 base_model.mm_projector")
                    elif found_location == 'base_model.model.mm_projector':
                        self.base_model.model.mm_projector = new_mm_projector
                        print(f"✓ 已更新 base_model.model.mm_projector")
                    elif found_location == 'base_model.base_model.mm_projector':
                        self.base_model.base_model.mm_projector = new_mm_projector
                        print(f"✓ 已更新 base_model.base_model.mm_projector")
                    
                    print(f"投影层注入完成，替换了 {replaced_count} 个Linear层")
                    
                elif isinstance(mm_projector, nn.Linear):
                    # 单个Linear层
                    layer_name = "mm_projector"
                    print(f"  替换投影层: {layer_name}")
                    
                    lora_layer = LoRALayer(
                        mm_projector,
                        r=self.lora_config.get('mm_projector_r', 8),
                        alpha=self.lora_config.get('mm_projector_alpha', 16),
                        dropout=self.lora_config.get('dropout', 0.1)
                    )
                    
                    # 根据找到的位置设置回去
                    if found_location == 'base_model.mm_projector':
                        self.base_model.mm_projector = lora_layer
                    elif found_location == 'base_model.model.mm_projector':
                        self.base_model.model.mm_projector = lora_layer
                    elif found_location == 'base_model.base_model.mm_projector':
                        self.base_model.base_model.mm_projector = lora_layer
                    
                    self.lora_layers[layer_name] = lora_layer
                    print(f"✓ 投影层注入完成")
                else:
                    print(f"⚠️ 不支持的投影层类型: {type(mm_projector)}")
            else:
                print("✗ 在所有可能的位置都未找到投影层")
        else:
            print("跳过投影层注入：mm_projector=False")
        
        # 最终统计
        print("\n----- 注入结果统计 -----")
        llm_count = sum(1 for k in self.lora_layers if k.startswith('llm.'))
        mm_count = sum(1 for k in self.lora_layers if k.startswith('mm_projector'))
        print(f"总替换层数: {len(self.lora_layers)}")
        print(f"  - LLM层 (16-31层的v_proj): {llm_count}")
        print(f"  - 投影层: {mm_count}")
        
        # 打印所有替换的LLM层
        if llm_count > 0:
            print(f"\n替换的LLM层详情:")
            for name in sorted(self.lora_layers.keys()):
                if name.startswith('llm.'):
                    print(f"  - {name}")
        print("------------------------\n")
    
    def set_lora_active(self, active=True):
        """
        控制所有LoRA层的激活状态
        """
        for layer in self.lora_layers.values():
            layer.use_lora = active
    
    def forward(self, *args, use_lora=True, output_hidden_states=False, **kwargs):
        """
        前向传播，可以选择是否使用LoRA
        """
        # 保存原始状态并设置新的use_lora值
        original_states = {}
        for name, layer in self.lora_layers.items():
            original_states[name] = layer.use_lora
            layer.use_lora = use_lora
        
        # 确保output_hidden_states被传递
        if 'output_hidden_states' not in kwargs:
            kwargs['output_hidden_states'] = output_hidden_states
        
        # 正常前向
        outputs = self.base_model(*args, **kwargs)
        
        # 恢复状态
        for name, layer in self.lora_layers.items():
            layer.use_lora = original_states[name]
        
        return outputs
    
    def get_hidden_states_with_lora_status(self, *args, layer_idx=-1, **kwargs):
        """
        获取指定层的hidden state，同时返回有无LoRA的版本
        """
        # 有LoRA的前向
        outputs_with = self.forward(
            *args, 
            use_lora=True, 
            output_hidden_states=True, 
            **kwargs
        )
        h_with = outputs_with.hidden_states[layer_idx]
        
        # 无LoRA的前向
        outputs_without = self.forward(
            *args, 
            use_lora=False, 
            output_hidden_states=True, 
            **kwargs
        )
        h_without = outputs_without.hidden_states[layer_idx]
        
        return h_with, h_without
    
    def merge_all_lora(self):
        """
        将所有LoRA参数合并回主模型（包括LLM和投影层）
        """
        print("开始合并LoRA参数...")
        merged_count = 0
        
        for name, layer in self.lora_layers.items():
            if hasattr(layer, 'merge_lora'):
                layer.merge_lora()
                merged_count += 1
        
        print(f"LoRA合并完成，共合并 {merged_count} 个层")
    
    def get_model(self):
        """
        兼容原有接口
        """
        return self.base_model.get_model()