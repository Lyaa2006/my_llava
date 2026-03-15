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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.train.llava_trainer import LLaVATrainer

# 导入持续学习组件
from peft.continual_trainer import ContinualLoRATrainer
from peft.llava_lora import LLaVAWithLoRA


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/mnt/lyaa/MCITlib/llava-v1.5-7b")
    vision_tower: str = field(default="/mnt/lyaa/MCITlib/clip-vit-large-patch14-336")
    version: str = field(default="v1")
    mm_vision_select_layer: int = field(default=-2)
    mm_vision_select_feature: str = field(default="patch")
    mm_projector_type: str = field(default='mlp2x_gelu')
    mm_patch_merge_type: str = field(default="flat")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)


@dataclass
class DataArguments:
    task_name: str = field(default="ArxivQA")
    data_path: str = field(default="/mnt/lyaa/MCITlib/UCIT/ArxivQA/train_4w.json")
    image_folder: str = field(default="/mnt/lyaa/MCITlib/UCIT/datasets")
    image_aspect_ratio: str = field(default="pad")
    is_multimodal: bool = field(default=True)
    image_processor: Optional[any] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: ['v_proj'])
    mm_projector_lora: bool = field(default=True)
    mm_projector_lora_r: int = field(default=8)
    mm_projector_lora_alpha: int = field(default=16)
    loss_hidden_weight: float = field(default=0.2)
    loss_text_qa_weight: float = field(default=0.3)
    loss_vision_qa_weight: float = field(default=0.5)
    hidden_layer_idx: int = field(default=-10)
    description_prompt: str = field(default="Please describe this image in detail.")
    task_sequence: List[str] = field(default_factory=lambda: ["ArxivQA", "CLEVR", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"])
    description_cache_dir: str = field(default="/mnt/lyaa/MCITlib/description_cache")
    save_safetensors :bool=False


class ContinualDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, data_args: DataArguments):
        super().__init__()
        self.data = []
        # 增加逻辑判断：如果路径为空或者是 dummy 状态，不执行文件读取
        if data_path and os.path.exists(data_path):
            with open(data_path, "r") as f:
                self.data = json.load(f)
                self.data=self.data[:2]
        else:
            print(f"WARNING: Data path {data_path} not found or empty. Initializing empty dataset.")

        print(f"DEBUG: 当前任务样本数: {len(self.data)}")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = data_args.image_folder
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (336, 336), color='white')

        conversations = item['conversations']
        question = conversations[0]['value']
        answer = conversations[1]['value']
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        labels = input_ids.clone()
        assistant_token = self.tokenizer.encode(conv.roles[1] + ":", add_special_tokens=False)
        if len(assistant_token) == 1:
            assistant_positions = (input_ids == assistant_token[0]).nonzero(as_tuple=True)[0]
            if len(assistant_positions) > 0:
                labels[:assistant_positions[-1] + 1] = IGNORE_INDEX

        return {
            'input_ids': input_ids,
            'labels': labels,
            'image': image,
            'question': question,
            'answer': answer,
            'image_path': item['image']
        }


class DataCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, image_processor: Optional[any] = None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        if not instances: return {}
        input_ids = [inst['input_ids'] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [inst['labels'] for inst in instances]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        images = [inst['image'] for inst in instances]

        if self.image_processor is not None:
            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        else:
            image_tensor = images

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'images': image_tensor,
            'questions': [inst['question'] for inst in instances],
            'answers': [inst['answer'] for inst in instances],
            'image_paths': [inst['image_path'] for inst in instances]
        }


def create_global_trainer(model, tokenizer, training_args, lora_config, loss_weights, image_processor, first_task_path):
    """
    创建全局的trainer实例。
    修复点：传入第一个任务的真实路径以防 dataset 初始化失败。
    """
    print("创建全局持续学习训练器...")
    
    dummy_data_args = DataArguments(
        task_name="init_dummy",
        data_path=first_task_path, 
        image_folder="",
        image_processor=image_processor
    )
    
    # 使用第一个任务的路径进行初始化，或者传空字符串（配合修改后的ContinualDataset）
    dummy_dataset = ContinualDataset(
        data_path=first_task_path,
        tokenizer=tokenizer,
        data_args=dummy_data_args
    )
    
    data_module = {
        'train_dataset': dummy_dataset,
        'eval_dataset': None,
        'data_collator': DataCollator(tokenizer=tokenizer, image_processor=image_processor)
    }

    trainer = ContinualLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        lora_config=lora_config,
        loss_weights=loss_weights,
        layer_idx=training_args.hidden_layer_idx,
        description_prompt=training_args.description_prompt,
        description_cache_dir=training_args.description_cache_dir,
        **data_module
    )
    
    return trainer


def train_task_with_trainer(trainer, task_name, data_path, image_folder, output_dir, tokenizer, image_processor):
    print(f"\n{'='*50}")
    print(f"开始训练任务: {task_name}")
    print(f"{'='*50}")
    
    data_args = DataArguments(
        task_name=task_name,
        data_path=data_path,
        image_folder=image_folder,
        image_processor=image_processor
    )
    
    dataset = ContinualDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        data_args=data_args
    )
    
    # 动态更新 trainer 内部的状态
    trainer.train_dataset = dataset
    trainer.data_collator = DataCollator(tokenizer=tokenizer, image_processor=image_processor)
    trainer.args.output_dir = output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    resume_from_checkpoint = list(pathlib.Path(output_dir).glob("checkpoint-*")) != []
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.on_task_end()
    
    
    
    return trainer.model


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32))

    print("加载基础模型...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        model_args.mm_vision_tower = model_args.vision_tower
        model.get_model().initialize_vision_modules(model_args=model_args)
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(dtype=compute_dtype)
        image_processor = vision_tower.image_processor
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else:
        image_processor = None

    # 数据路径映射
    data_paths = {
        "ArxivQA": "/mnt/lyaa/MCITlib/UCIT/ArxivQA/train_4w.json",
        "CLEVR": "/mnt/lyaa/MCITlib/UCIT/CLEVR/train_4w.json",
        "Flickr30k": "/mnt/lyaa/MCITlib/UCIT/Flickr30k/train_brief_4w.json",
        "IconQA": "/mnt/lyaa/MCITlib/UCIT/IconQA/train.json",
        "ImageNet-R": "/mnt/lyaa/MCITlib/UCIT/ImageNet-R/train.json",
        "VizWiz": "/mnt/lyaa/MCITlib/UCIT/VizWiz/train.json"
    }
    
    image_folders = {task: "/mnt/lyaa/MCITlib/UCIT/datasets" for task in data_paths.keys()}

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

    loss_weights = {
        'hidden': training_args.loss_hidden_weight,
        'text_qa': training_args.loss_text_qa_weight,
        'vision_qa': training_args.loss_vision_qa_weight
    }

    # 获取序列中第一个任务的路径，用于安全初始化
    first_task_name = training_args.task_sequence[0] if training_args.task_sequence else None
    first_path = data_paths.get(first_task_name, "")

    # 创建全局 trainer
    global_trainer = create_global_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        lora_config=lora_config,
        loss_weights=loss_weights,
        image_processor=image_processor,
        first_task_path=first_path
    )
    
    print(f"\n开始持续学习流程，共 {len(training_args.task_sequence)} 个任务...")
    
    for task_idx, task_name in enumerate(training_args.task_sequence):
        data_path = data_paths.get(task_name)
        image_folder = image_folders.get(task_name, "/mnt/lyaa/MCITlib/UCIT/datasets")
        output_dir = f"/mnt/lyaa/MCITlib/checkpoints/continual/{task_name}"
        
        train_task_with_trainer(
            trainer=global_trainer,
            task_name=task_name,
            data_path=data_path,
            image_folder=image_folder,
            output_dir=output_dir,
            tokenizer=tokenizer,
            image_processor=image_processor
        )
        
        print(f"任务 {task_name} 完成，进度: {task_idx+1}/{len(training_args.task_sequence)}")
    
    
    print("\n所有任务训练完成！")


if __name__ == "__main__":
    main()