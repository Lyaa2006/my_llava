import torch

import torch.nn as nn

import torch.nn.functional as F

from transformers import Trainer

import os



class ContinualLoRATrainer(Trainer):

    """

    重构版支持持续学习的 Trainer

    改进：移除不可靠的缓存机制，增强多图片批处理的稳定性

    """

    def __init__(self,

                 lora_config=None,

                 loss_weights=None,

                 layer_idx=-10,

                 description_prompt=None,

                 description_cache_dir=None,

                 **kwargs):

       

        self.description_prompt = description_prompt or "Please describe this image in detail"

        self.layer_idx = layer_idx

        self.description_cache_dir = description_cache_dir or "./description_cache"

        os.makedirs(self.description_cache_dir, exist_ok=True)

       

        self.loss_weights = loss_weights or {

            'hidden': 0.2,

            'text_qa': 0.3,

            'vision_qa': 0.5

        }

       

        super().__init__(**kwargs)

       

        self.lora_config = lora_config or {

            'r': 8,

            'alpha': 16,

            'target_modules': ['v_proj'],

            'mm_projector': True,

            'dropout': 0.05

        }

       

        # 包装模型

        from peft.llava_lora import LLaVAWithLoRA

        if not isinstance(self.model, LLaVAWithLoRA):

            print("正在用 LoRA 包装模型...")

            self.model = LLaVAWithLoRA(self.model, self.lora_config)

        else:

            print("模型已经是 LLaVAWithLoRA 包装状态。")



        if hasattr(self.model, "gradient_checkpointing_enable"):

            self.model.gradient_checkpointing_enable()

           

        self.description_cache = {}

        self._load_cache()



    def _get_cache_path(self, image_path):

        safe_name = image_path.replace('/', '_').replace('\\', '_')

        return os.path.join(self.description_cache_dir, f"{safe_name}.txt")



    def _load_cache(self):

        if os.path.exists(self.description_cache_dir):

            for f in os.listdir(self.description_cache_dir):

                if f.endswith('.txt'):

                    cache_path = os.path.join(self.description_cache_dir, f)

                    with open(cache_path, 'r', encoding='utf-8') as file:

                        image_key = f[:-4]

                        self.description_cache[image_key] = file.read()

            print(f"加载了 {len(self.description_cache)} 条 description 缓存")



    def _save_description_to_cache(self, image_path, description):

        cache_path = self._get_cache_path(image_path)

        with open(cache_path, 'w', encoding='utf-8') as f:

            f.write(description)

        safe_key = image_path.replace('/', '_').replace('\\', '_')

        self.description_cache[safe_key] = description



    @torch.no_grad()

    def generate_description(self, model, image, image_path=None):

        """离线生成描述，强制切换模式"""

        if image_path is not None:

            safe_key = image_path.replace('/', '_').replace('\\', '_')

            if safe_key in self.description_cache:

                return self.description_cache[safe_key]

       

        orig_mode = model.training

        model.eval()

        if hasattr(model, 'set_lora_active'):

            model.set_lora_active(False)

       

        prompt = f"USER: {self.description_prompt}\nASSISTANT:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)



        generated_ids = model.generate(

            inputs=inputs['input_ids'],

            images=image.unsqueeze(0) if image is not None else None,

            max_new_tokens=100,

            do_sample=False

        )

       

        description = self.tokenizer.decode(

            generated_ids[0][inputs['input_ids'].shape[1]:],

            skip_special_tokens=True

        ).strip()

       

        if hasattr(model, 'set_lora_active'):

            model.set_lora_active(True)

        model.train(orig_mode)

       

        if image_path is not None:

            self._save_description_to_cache(image_path, description)

        return description



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        target_dtype = next(model.parameters()).dtype

        device = next(model.parameters()).device

        print(inputs)

        exit(0)

        images = inputs.get('images')

        if images is not None:

            images = images.to(device=device, dtype=target_dtype)

           

        input_ids = inputs.get('input_ids')

        labels = inputs.get('labels')

        image_paths = inputs.get('image_paths')



        # 检查是否使用了 DataParallel 等包装

        raw_model = model.module if hasattr(model, 'module') else model



        # --- 步骤 1: 实时获取 Base Model 的 Hidden State (去缓存版) ---

        with torch.no_grad():

            # 记录当前模式

            orig_training_mode = raw_model.training

            raw_model.eval() # 切换到 eval 模式以获得稳定的特征

           

            if hasattr(raw_model, 'set_lora_active'):

                raw_model.set_lora_active(False) # 临时禁用 LoRA

           

            outputs_base = raw_model(

                input_ids=input_ids,

                images=images,

                output_hidden_states=True,

                return_dict=True

            )

            # 提取基准特征并断开梯度

            h_without_lora = outputs_base.hidden_states[self.layer_idx].mean(dim=1).detach()

           

            # 恢复训练模式

            if hasattr(raw_model, 'set_lora_active'):

                raw_model.set_lora_active(True)

            raw_model.train(orig_training_mode)



        # --- 步骤 2: 多模态前向传播 (带梯度) ---

        # 显式确保 LoRA 激活

        if hasattr(raw_model, 'set_lora_active'):

            raw_model.set_lora_active(True)



        outputs_vision = model(

            input_ids=input_ids,

            images=images,

            labels=labels,

            output_hidden_states=True,

            return_dict=True

        )

        loss_vision = outputs_vision.loss

        h_with_lora = outputs_vision.hidden_states[self.layer_idx].mean(dim=1)

       

        # 计算 Hidden 损失

        loss_hidden = torch.nn.functional.mse_loss(h_with_lora.float(), h_without_lora.float())



        # --- 步骤 3: 文本路径损失 ---

        loss_text = torch.tensor(0.0).to(device)

        if images is not None and image_paths is not None and 'questions' in inputs:

            # 提取对话中的回答

            answers_text = inputs.get('answers', None)

            if answers_text is None and labels is not None:

                answers_text = []

                for label_tensor in labels:

                    valid_ids = label_tensor[label_tensor >= 0]

                    decoded_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)

                    answers_text.append(decoded_text)

           

            if answers_text and len(answers_text) > 0:

                # 实时生成图片描述

                descriptions = [self.generate_description(raw_model, images[i], img_path)

                               for i, img_path in enumerate(image_paths)]

               

                text_input_ids, text_labels = self._build_text_inputs_and_labels(

                    descriptions,

                    inputs['questions'],

                    answers_text,

                    device

                )

               

                # 文本路径前向传播 (绕过多模态 Projector)

                from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

                outputs_text = super(LlavaLlamaForCausalLM, raw_model).forward(

                    input_ids=text_input_ids,

                    labels=text_labels,

                    images=None,

                    attention_mask=None

                )

                loss_text = outputs_text.loss if outputs_text.loss is not None else loss_text

           

        # 损失加权汇总

        total_loss = (

           # self.loss_weights['hidden'] * loss_hidden +

            self.loss_weights['text_qa'] * loss_text +

            self.loss_weights['vision_qa'] * loss_vision

        )



        # 调试打印 (建议每步打印直到确认收敛)

        if self.state.global_step % 1 == 0:

            print(f"Step {self.state.global_step} | Total Loss: {total_loss.item():.4f} | Hidden: {loss_hidden.item():.4f} | Vision: {loss_vision.item():.4f} | Text: {loss_text.item():.4f}")



        return (total_loss, outputs_vision) if return_outputs else total_loss



    def _build_text_inputs_and_labels(self, descs, questions, answers, device):

        tokenizer = self.tokenizer

        full_input_ids = []

        full_labels = []



        for d, q, a in zip(descs, questions, answers):

            prompt = f"USER: Description: {d}\nQuestion: {q}\nASSISTANT:"

            full_text = f"{prompt} {a}{tokenizer.eos_token}"

           

            input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]

           

            labels = input_ids.clone()

            labels[:len(prompt_ids)] = -100 # Mask 掉 Prompt

           

            full_input_ids.append(input_ids)

            full_labels.append(labels)



        # Padding

        text_input_ids = torch.nn.utils.rnn.pad_sequence(

            full_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id

        ).to(device)

       

        text_labels = torch.nn.utils.rnn.pad_sequence(

            full_labels, batch_first=True, padding_value=-100

        ).to(device)



        return text_input_ids, text_labels



    def on_task_end(self):

        """任务结束处理：合并 LoRA 并清理资源"""

        print("\n" + "="*30 + " 任务结束：合并并重置 LoRA " + "="*30)

       

        if hasattr(self.model, 'merge_all_lora'):

            self.model.merge_all_lora()

           

        if hasattr(self.model, 'reset_lora'):

            self.model.reset_lora()

           

        torch.cuda.empty_cache()

        print("="*80 + "\n")