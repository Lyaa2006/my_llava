import os
import sys
import json
import torch
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List
from PIL import Image

import transformers
from torch.utils.data import Dataset

# 添加项目路径
sys.path.append('/mnt/lyaa/MCITlib/LLaVA/my')

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.train.llava_trainer import LLaVATrainer

# 导入我们之前写的持续学习组件
from peft.continual_trainer import ContinualLoRATrainer
from peft.llava_lora import LLaVAWithLoRA


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/mnt/lyaa/MCITlib/llava-v1.5-7b")
    vision_tower: str = field(default="/mnt/lyaa/MCITlib/clip-vit-large-patch14-336")
    version: str = field(default="v1")
    
    # 视觉相关参数
    mm_vision_select_layer: int = field(default=-2)  # 选择哪一层的特征
    mm_vision_select_feature: str = field(default="patch")  # 选择什么特征
    mm_projector_type: str = field(default='mlp2x_gelu')  # 视觉投影器类型
    mm_patch_merge_type: str = field(default="flat")  # patch合并方式
    
    # 可以添加其他常用参数
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)  # 预训练的视觉投影器
    mm_use_im_start_end: bool = field(default=False)  # 是否使用图像开始/结束标记
    mm_use_im_patch_token: bool = field(default=False)  # 是否使用图像patch标记

@dataclass
class DataArguments:
    """数据参数 - 支持多个任务的训练"""
    task_name: str = field(default="ArxivQA")  # 当前任务名
    data_path: str = field(default="/mnt/lyaa/MCITlib/UCIT/ArxivQA/train_4w.json")
    image_folder: str = field(default="/mnt/lyaa/MCITlib/UCIT/datasets")
    image_aspect_ratio: str = field(default="pad")
    is_multimodal: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # 基础训练参数
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    
    # LoRA参数
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: ['v_proj'])
    
    mm_projector_lora: bool = field(default=True)  # 是否给投影层加LoRA
    mm_projector_lora_r: int = field(default=8)    # 投影层LoRA秩
    mm_projector_lora_alpha: int = field(default=16)  # 投影层LoRA alpha
    # 持续学习loss权重
    loss_hidden_weight: float = field(default=0.2)   # hidden state一致性
    loss_text_qa_weight: float = field(default=0.3)  # 纯文本QA（description+question）
    loss_vision_qa_weight: float = field(default=0.5) # 多模态QA
    
    # hidden state对比的层
    hidden_layer_idx: int = field(default=-10)
    
    # description生成参数
    description_prompt: str = field(
        default="Please describe this image in detail, including any text, objects, and their relationships."
    )
    
    # 任务序列（用于持续学习）
    task_sequence: List[str] = field(default_factory=lambda: [
        "ArxivQA", "CLEVR", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"
    ])
    description_cache_dir: str = field(
        default="/mnt/lyaa/MCITlib/description_cache"
    )


class ContinualDataset(Dataset):
    """
    持续学习数据集 - 支持从JSON文件加载
    """
    def __init__(self, data_path: str, tokenizer, data_args: DataArguments):
        super().__init__()
        
        # 加载JSON
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = data_args.image_folder
        
        # 设置对话模板
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        
        print(f"加载数据集: {data_path}")
        print(f"样本数量: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 加载图像
        image_path = os.path.join(self.image_folder, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            # 返回一个空图像
            image = Image.new('RGB', (336, 336), color='white')
        
        # 2. 处理对话
        conversations = item['conversations']
        
        # 提取问题和答案
        human_conv = conversations[0]
        gpt_conv = conversations[1]
        
        assert human_conv['from'] == 'human'
        assert gpt_conv['from'] == 'gpt'
        
        # 问题文本（包含<image>标记）
        question = human_conv['value']
        answer = gpt_conv['value']
        
        # 3. 构建对话格式
        conv = conversation_lib.default_conversation.copy()
        
        # 添加human消息（包含图像标记）
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        
        # 获取prompt
        prompt = conv.get_prompt()
        
        # 4. 编码
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        
        # 5. 创建labels（只有答案部分参与loss计算）
        labels = input_ids.clone()
        
        # 找到答案开始的位置（在"ASSISTANT:"之后）
        sep = conv.sep + conv.roles[1] + ": "
        total_len = input_ids.size(0)
        
        # 简单方法：human部分设置为IGNORE_INDEX
        # 更精确的方法需要解析对话，这里简化处理
        # 实际使用时可以根据需要优化
        assistant_token = self.tokenizer.encode(conv.roles[1] + ":", add_special_tokens=False)
        if len(assistant_token) == 1:
            # 找到assistant token的位置
            assistant_positions = (input_ids == assistant_token[0]).nonzero(as_tuple=True)[0]
            if len(assistant_positions) > 0:
                # 最后一个assistant token之后是答案
                labels[:assistant_positions[-1] + 1] = IGNORE_INDEX
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'image': image,
            'question': question,  # 保存原始问题，用于生成description
            'answer': answer,
            'image_path': item['image']  # 用作key
        }


@dataclass
class DataCollator:
    """数据整理器"""
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict:
        # 整理input_ids
        input_ids = [inst['input_ids'] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        # 整理labels
        labels = [inst['labels'] for inst in instances]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        
        # 整理图像
        images = [inst['image'] for inst in instances]
        
        # 整理其他信息
        questions = [inst['question'] for inst in instances]
        answers = [inst['answer'] for inst in instances]
        image_paths = [inst['image_path'] for inst in instances]
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'images': images,  # 将在预处理中转换为tensor
            'questions': questions,
            'answers': answers,
            'image_paths': image_paths  # 用作key
        }
        
        return batch


def train_task(model_args, data_args, training_args, tokenizer, model=None):
    """
    训练单个任务
    """
    print(f"\n{'='*60}")
    print(f"开始训练任务: {data_args.task_name}")
    print(f"数据路径: {data_args.data_path}")
    print(f"{'='*60}\n")
    training_args.local_rank = -1
    training_args.ddp_find_unused_parameters = False
    training_args.dataloader_drop_last = True
    if model is not None:
        device = next(model.parameters()).device
        print(f"模型当前在设备: {device}")
    # 1. 准备数据集
    dataset = ContinualDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args
    )
    
    data_collator = DataCollator(tokenizer=tokenizer)
    
    # 2. 准备数据模块
    data_module = {
        'train_dataset': dataset,
        'eval_dataset': None,
        'data_collator': data_collator
    }
    
    # 3. 配置LoRA
    lora_config = {
        'r': training_args.lora_r,
        'alpha': training_args.lora_alpha,
        'target_modules': training_args.lora_target_modules,
        'target_layers': list(range(16, 32)), 
        'dropout': training_args.lora_dropout,
        'mm_projector': training_args.mm_projector_lora,
        'mm_projector_r': training_args.mm_projector_lora_r,
        'mm_projector_alpha': training_args.mm_projector_lora_alpha
    }
    
    # 4. 配置loss权重
    loss_weights = {
        'hidden': training_args.loss_hidden_weight,
        'text_qa': training_args.loss_text_qa_weight,
        'vision_qa': training_args.loss_vision_qa_weight
    }
    
    # 5. 创建trainer
    trainer = ContinualLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        lora_config=lora_config,
        loss_weights=loss_weights,
        layer_idx=training_args.hidden_layer_idx,
        description_prompt=training_args.description_prompt,
        description_cache_dir=training_args.description_cache_dir,  # 新增
        **data_module
    )
    
    # 6. 训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        model.to('cuda')
        trainer.train()
    
    # 7. 任务结束，合并LoRA
    trainer.on_task_end()
    
    # 8. 保存模型
    save_path = os.path.join(training_args.output_dir, f"task_{data_args.task_name}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"任务 {data_args.task_name} 完成，模型已保存到 {save_path}")
    
    return trainer.model  # 返回更新后的模型


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载基础模型（第一次运行时）
    print("加载基础模型...")
    
    # 创建模型配置
    if model_args.vision_tower is not None:
        # 加载LLaVA模型
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    
    model.config.use_cache = False
    
    # 2. 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 3. 初始化视觉模块（如果有）
    if model_args.vision_tower is not None:
        print("初始化视觉模块...")
        
        # 设置vision tower参数
        model_args.mm_vision_tower = model_args.vision_tower
        model_args.tune_mm_mlp_adapter = False  # 不训练vision tower
        
        # 初始化
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp if hasattr(training_args, 'fsdp') else None
        )
        
        # 加载vision tower权重
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        
        # 移动到设备
        vision_tower.to(device=device, dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
        
        # 设置图像处理器
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        
        # 更新模型配置
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_use_im_start_end = False
        model.config.mm_use_im_patch_token = False
        
        # 初始化vision tokenizer
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    
    # 4. 持续学习：按顺序训练所有任务
    task_sequence = training_args.task_sequence
    
    for i, task_name in enumerate(task_sequence):
        print(f"\n{'#'*60}")
        print(f"任务 {i+1}/{len(task_sequence)}: {task_name}")
        print(f"{'#'*60}\n")
        
        # 设置当前任务的数据路径
        if task_name == "ArxivQA":
            data_args.data_path = "/mnt/lyaa/MCITlib/UCIT/ArxivQA/train_4w.json"
        elif task_name == "CLEVR":
            data_args.data_path = "/mnt/lyaa/MCITlib/UCIT/CLEVR/train_4w.json"
        elif task_name == "Flickr30k":
            data_args.data_path = "/mnt/lyaa/MCITlib/UCIT/Flickr30k/train_brief_4w.json"
        elif task_name == "IconQA":
            data_args.data_path = "/mnt/lyaa/MCITlib/UCIT/IconQA/train.json"
        elif task_name == "ImageNet-R":
            data_args.data_path = "/mnt/lyaa/MCITlib/UCIT/ImageNet-R/train.json"
        elif task_name == "VizWiz":
            data_args.data_path = "/mnt/lyaa/MCITlib/UCIT/VizWiz/train.json"
        
        data_args.task_name = task_name
        
        # 设置输出目录
        training_args.output_dir = f"/mnt/lyaa/MCITlib/checkpoints/continual/{task_name}"
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 训练当前任务
        model = train_task(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
            model=model  # 传递上一个任务训练后的模型
        )


if __name__ == "__main__":
    main()