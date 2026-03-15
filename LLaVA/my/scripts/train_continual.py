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



# 导入我们之前写的持续学习组件

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



class ContinualDataset(Dataset):

    def __init__(self, data_path: str, tokenizer, data_args: DataArguments):

        super().__init__()

        with open(data_path, "r") as f:

            self.data = json.load(f)

        self.data = self.data[:1]

        print(f"DEBUG: 任务数据已截断，当前任务样本数: {len(self.data)}")

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



@dataclass

@dataclass

class DataCollator:

    tokenizer: transformers.PreTrainedTokenizer

    image_processor: Optional[any] = None  # 新增：接收图像处理器



    def __call__(self, instances: Sequence[Dict]) -> Dict:

        input_ids = [inst['input_ids'] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(

            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id

        )

        labels = [inst['labels'] for inst in instances]

        labels = torch.nn.utils.rnn.pad_sequence(

            labels, batch_first=True, padding_value=IGNORE_INDEX

        )

       

        # --- 核心修改：将 PIL Images 转换为 Tensor ---

        images = [inst['image'] for inst in instances]

        if self.image_processor is not None:

            # 使用 LLaVA 的图像处理器进行缩放、归一化等预处理

            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']

        else:

            # 这种方式不推荐，因为没经过模型预处理，效果会差

            image_tensor = images



        return {

            'input_ids': input_ids,

            'labels': labels,

            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),

            'images': image_tensor,  # 现在是 Tensor 了

            'questions': [inst['question'] for inst in instances],

            'image_paths': [inst['image_path'] for inst in instances]

        }



def train_task(model_args, data_args, training_args, tokenizer, model=None):

    print(f"\n开始训练任务: {data_args.task_name}")

   

    dataset = ContinualDataset(data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args)

    data_module = {'train_dataset': dataset, 'eval_dataset': None, 'data_collator': DataCollator(tokenizer=tokenizer,image_processor=data_args.image_processor)}

   

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

   

    # 修改：移除显式的 model.to('cuda')，Trainer 会根据分布式环境自动处理

    trainer.train(resume_from_checkpoint=list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) != [])

   

    trainer.on_task_end()

    save_path = os.path.join(training_args.output_dir, f"task_{data_args.task_name}")

    trainer.save_model(save_path)

    return trainer.model



def main():

    # 显存碎片优化方案

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

   

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

   

    # 根据分布式环境选择精度

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

        vision_tower.to(dtype=compute_dtype) # 移除 .to(device)，让 DeepSpeed/DDP 分配

       

        data_args.image_processor = vision_tower.image_processor

        model.config.image_aspect_ratio = data_args.image_aspect_ratio

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)



    # 核心修改：不再在这里写循环，而是根据环境变量管理任务

    # 但为了兼容你原本的逻辑，保持循环，但移除内部的强制 device 转换

    task_sequence = training_args.task_sequence

   

    # 路径映射逻辑

    data_paths = {

        "ArxivQA": "/mnt/lyaa/MCITlib/UCIT/ArxivQA/train_4w.json",

        "CLEVR": "/mnt/lyaa/MCITlib/UCIT/CLEVR/train_4w.json",

        "Flickr30k": "/mnt/lyaa/MCITlib/UCIT/Flickr30k/train_brief_4w.json",

        "IconQA": "/mnt/lyaa/MCITlib/UCIT/IconQA/train.json",

        "ImageNet-R": "/mnt/lyaa/MCITlib/UCIT/ImageNet-R/train.json",

        "VizWiz": "/mnt/lyaa/MCITlib/UCIT/VizWiz/train.json"

    }



    for task_name in task_sequence:

        data_args.data_path = data_paths.get(task_name)

        data_args.task_name = task_name

        training_args.output_dir = f"/mnt/lyaa/MCITlib/checkpoints/continual/{task_name}"

        os.makedirs(training_args.output_dir, exist_ok=True)

       

        model = train_task(model_args, data_args, training_args, tokenizer, model=model)
        
if __name__ == "__main__":

    main()