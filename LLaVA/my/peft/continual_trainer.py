import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import os
import json

class ContinualLoRATrainer(Trainer):
    """
    支持持续学习的Trainer，包含三部分loss
    
    损失函数构成：
    1. hidden_loss: 加LoRA后模型的hidden state与不加LoRA的hidden state尽量相似
    2. text_qa_loss: question + description -> answer (使用自然语言description)
    3. vision_qa_loss: image + question -> answer (多模态路径)
    
    优化：description以自然语言形式保存，避免重复计算
    """
    def __init__(self, 
                 lora_config=None, 
                 loss_weights=None, 
                 layer_idx=-10,
                 description_prompt=None,
                 description_cache_dir=None,  # 新增：description缓存目录
                 **kwargs):
        
        # 保存参数
        self.description_prompt = description_prompt or "Please describe this image in detail"
        self.layer_idx = layer_idx
        self.description_cache_dir = description_cache_dir or "./description_cache"
        
        # 创建缓存目录
        os.makedirs(self.description_cache_dir, exist_ok=True)
        
        self.loss_weights = loss_weights or {
            'hidden': 0.2,
            'text_qa': 0.3,
            'vision_qa': 0.5
        }
        
        # 先初始化Trainer
        super().__init__(**kwargs)
        
        # LoRA配置
        self.lora_config = lora_config or {
            'r': 8, 
            'alpha': 16, 
            'target_modules': ['v_proj'],
            'mm_projector': True,
            'dropout': 0.05
        }
        
        # 用LoRA包装模型
        print("正在用LoRA包装模型...")
        from peft.llava_lora import LLaVAWithLoRA
        self.model = LLaVAWithLoRA(self.model, self.lora_config)
        print("模型包装完成")
        
        # description缓存字典
        self.description_cache = {}
        self._load_cache()
    
    def _get_cache_path(self, image_path):
        """获取缓存文件路径"""
        # 将image_path转换为安全的文件名
        safe_name = image_path.replace('/', '_').replace('\\', '_')
        return os.path.join(self.description_cache_dir, f"{safe_name}.txt")
    
    def _load_cache(self):
        """加载已有的description缓存"""
        if os.path.exists(self.description_cache_dir):
            for f in os.listdir(self.description_cache_dir):
                if f.endswith('.txt'):
                    cache_path = os.path.join(self.description_cache_dir, f)
                    with open(cache_path, 'r') as file:
                        image_key = f[:-4]  # 去掉.txt后缀
                        self.description_cache[image_key] = file.read()
            print(f"加载了 {len(self.description_cache)} 条description缓存")
    
    def _save_description_to_cache(self, image_path, description):
        """保存description到缓存文件"""
        cache_path = self._get_cache_path(image_path)
        with open(cache_path, 'w') as f:
            f.write(description)
        # 同时更新内存缓存
        safe_key = image_path.replace('/', '_').replace('\\', '_')
        self.description_cache[safe_key] = description
    
    def generate_description(self, model, image, image_path=None):
        """
        使用原始模型生成自然语言description
        只运行一次，结果会被缓存
        
        Args:
            model: 原始模型（不带LoRA）
            image: 输入图像
            image_path: 图像路径，用于缓存
        """
        # 如果有缓存且提供了image_path，直接返回缓存的description
        if image_path is not None:
            safe_key = image_path.replace('/', '_').replace('\\', '_')
            if safe_key in self.description_cache:
                return self.description_cache[safe_key]
        
        # 构建description prompt
        prompt = f"USER: {self.description_prompt}\nASSISTANT:"
        
        # 使用tokenizer处理prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        # 生成description
        with torch.no_grad():
            # 不使用LoRA生成
            if hasattr(model, 'set_lora_active'):
                model.set_lora_active(False)
            
            # 生成文本
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                images=image.unsqueeze(0) if image is not None else None,
                max_new_tokens=100,
                do_sample=False,
                num_beams=1
            )
            
            # 解码
            description = self.tokenizer.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # 恢复LoRA状态
            if hasattr(model, 'set_lora_active'):
                model.set_lora_active(True)
        
        # 保存到缓存
        if image_path is not None:
            self._save_description_to_cache(image_path, description)
        
        return description
    
    def get_hidden_state(self, model, input_ids, images, use_lora=True):
        """
        获取指定层的hidden state
        通过use_lora参数控制是否使用LoRA
        """
        device = next(model.parameters()).device
        
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if images is not None and isinstance(images, torch.Tensor) and images.device != device:
            images = images.to(device)
        
        with torch.no_grad() if not use_lora else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                images=images,
                output_hidden_states=True,
                use_lora=use_lora
            )
            hidden_states = outputs.hidden_states[self.layer_idx]
            description_hidden = hidden_states.mean(dim=1)
        
        return description_hidden
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算三部分loss
        """
        device = next(model.parameters()).device
        
        # 获取输入
        images = inputs.get('images')
        input_ids = inputs.get('input_ids')
        labels = inputs.get('labels')
        image_paths = inputs.get('image_paths')  # 用于缓存key
        
        # 将输入移动到正确的设备
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if labels is not None and labels.device != device:
            labels = labels.to(device)
        
        # 1. Hidden state对比损失（保持不变）
        h_without_lora = self.get_hidden_state(
            model, input_ids, images, use_lora=False
        ).detach()
        
        h_with_lora = self.get_hidden_state(
            model, input_ids, images, use_lora=True
        )
        
        loss_hidden = F.mse_loss(h_with_lora, h_without_lora)
        
        # 2. 多模态路径损失 (image + question -> answer)
        outputs_vision = model(
            input_ids=input_ids,
            images=images,
            labels=labels,
            use_lora=True,
            output_hidden_states=False
        )
        loss_vision = outputs_vision.loss
        
        # 3. 文本路径损失 (description + question -> answer)
        loss_text = torch.tensor(0.0).to(device)
        
        if images is not None and image_paths is not None:
            # 为batch中的每个图像生成description
            descriptions = []
            for i, img_path in enumerate(image_paths):
                # 使用原始模型（不带LoRA）生成description
                if isinstance(images, list):
                    # images是PIL图像列表
                    desc = self.generate_description(
                        self.model.base_model,  # 使用原始模型
                        images[i] if i < len(images) else None,
                        img_path
                    )
                else:
                    # images是tensor，需要处理
                    desc = self.generate_description(
                        self.model.base_model,
                        images[i] if images is not None else None,
                        img_path
                    )
                descriptions.append(desc)
            
            # 构建description+question的输入
            # 这里需要根据你的数据格式构造新的input_ids
            # 假设inputs中包含了原始问题questions
            if 'questions' in inputs:
                text_inputs = self._build_text_inputs(
                    descriptions, 
                    inputs['questions'],
                    device
                )
                
                # 计算文本路径的loss
                outputs_text = model(
                    input_ids=text_inputs,
                    images=None,  # 不输入图像
                    labels=labels,  # 注意：labels可能需要调整
                    use_lora=True,
                    output_hidden_states=False
                )
                loss_text = outputs_text.loss if outputs_text.loss is not None else loss_text
        
        # 加权组合
        total_loss = (
            self.loss_weights['hidden'] * loss_hidden +
            self.loss_weights['text_qa'] * loss_text +
            self.loss_weights['vision_qa'] * loss_vision
        )
        
        # 打印loss
        if self.state.global_step % 100 == 0:
            print(f"\nStep {self.state.global_step}:")
            print(f"  hidden_loss: {loss_hidden.item():.4f}")
            print(f"  text_qa_loss: {loss_text.item():.4f}")
            print(f"  vision_qa_loss: {loss_vision.item():.4f}")
            print(f"  total_loss: {total_loss.item():.4f}")
        
        return (total_loss, outputs_vision) if return_outputs else total_loss
    
    def _build_text_inputs(self, descriptions, questions, device):
        """
        构建纯文本输入：description + question
        """
        batch_texts = []
        for desc, q in zip(descriptions, questions):
            # 构建prompt：先描述图像，再问问题
            text = f"USER: The image shows: {desc}\nQuestion: {q}\nASSISTANT:"
            batch_texts.append(text)
        
        # tokenize
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(device)
        
        return inputs['input_ids']
    
    def on_task_end(self):
        """
        每个任务结束时调用
        """
        print(f"\n{'='*50}")
        print("任务结束，正在合并LoRA...")
        
        # 合并LoRA权重
        self.model.merge_all_lora()
        
        # 重置LoRA
        print("重置LoRA，准备下一个任务...")
        if hasattr(self.model, 'reset_lora'):
            self.model.reset_lora()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存description缓存索引（可选）
        cache_index = os.path.join(self.description_cache_dir, "cache_index.json")
        with open(cache_index, 'w') as f:
            json.dump(list(self.description_cache.keys()), f)
        
        print(f"任务完成，模型已更新，description缓存保存在: {self.description_cache_dir}")
        print(f"{'='*50}\n")