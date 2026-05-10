import os
import re
import torch
import torch.nn.functional as F

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            if self.args.is_SAT:
                lengths = self.train_dataset.modality_lengths_for_SAT
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _extract_description_states(self, model, inputs, compute_lora_orth=False):
        lora_losses = []
        hooks = []
        if compute_lora_orth and getattr(self.args, "orth_lora_weight", 0.0) > 0:
            targets = self._iter_lora_activation_targets(model)
            self._last_lora_orth_target_count = len(targets)

            def build_hook(module):
                def hook(module, hook_inputs, hook_outputs):
                    if not hook_inputs:
                        return
                    hidden = hook_inputs[0]
                    if hidden is None:
                        return

                    dropout = self._get_lora_dropout(module, "default")
                    if dropout is None:
                        dropout = torch.nn.Identity()

                    current_scale = self._get_lora_scaling(module, "default")
                    old_scale = self._get_lora_scaling(module, "frozen_old") * getattr(self.args, "old_lora_scale", 1.0)

                    current_input = dropout(hidden)
                    current_output = module.lora_B["default"](module.lora_A["default"](current_input)) * current_scale

                    with torch.no_grad():
                        old_input = dropout(hidden.detach())
                        old_output = module.lora_B["frozen_old"](module.lora_A["frozen_old"](old_input)) * old_scale

                    cosine = F.cosine_similarity(current_output.float(), old_output.float(), dim=-1)
                    lora_losses.append(cosine.pow(2).mean())

                return hook

            for module in targets:
                hooks.append(module.register_forward_hook(build_hook(module)))

        description_outputs = model(
            input_ids=inputs["description_input_ids"],
            attention_mask=inputs["description_attention_mask"],
            images=inputs.get("images"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        for hook in hooks:
            hook.remove()

        lora_orth_loss = torch.stack(lora_losses).mean() if lora_losses else None
        if compute_lora_orth:
            self._last_lora_orth_loss_count = len(lora_losses)
        hidden_states = description_outputs.hidden_states[self.args.description_hidden_layer]
        description_sequences = []
        key_mask_sequences = []
        description_key_mask = inputs.get("description_key_mask")
        lengths = inputs["description_attention_mask"].long().sum(dim=1).tolist()
        for batch_idx, cur_len in enumerate(lengths):
            start_idx = max(0, cur_len - self.args.description_max_tokens)
            description_sequences.append(hidden_states[batch_idx, start_idx:cur_len])
            if description_key_mask is not None:
                key_mask_sequences.append(description_key_mask[batch_idx, start_idx:cur_len])
        if description_key_mask is None:
            if compute_lora_orth:
                return description_sequences, None, lora_orth_loss
            return description_sequences, None
        if compute_lora_orth:
            return description_sequences, key_mask_sequences, lora_orth_loss
        return description_sequences, key_mask_sequences

    def _get_lora_dropout(self, module, adapter_name):
        dropout = getattr(module, "lora_dropout", None)
        if dropout is None:
            return None
        if isinstance(dropout, (dict, torch.nn.ModuleDict)):
            return dropout[adapter_name] if adapter_name in dropout else None
        return dropout

    def _get_lora_scaling(self, module, adapter_name):
        scaling = getattr(module, "scaling", 1.0)
        if isinstance(scaling, dict):
            return scaling.get(adapter_name, 1.0)
        return scaling

    def _iter_lora_activation_targets(self, model):
        total_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        if total_layers is None:
            base_model = getattr(model, "model", None)
            total_layers = getattr(getattr(base_model, "config", None), "num_hidden_layers", None)
        target_modules = getattr(self.args, "orth_lora_target_modules", "q_proj,v_proj,o_proj,down_proj")
        target_modules = {name.strip() for name in target_modules.split(",") if name.strip()}

        named_lora_modules = []
        layer_ids = []
        for name, module in model.named_modules():
            if not target_modules:
                break
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if "default" not in module.lora_A or "frozen_old" not in module.lora_A:
                continue
            if module.r.get("default", 0) <= 0 or module.r.get("frozen_old", 0) <= 0:
                continue
            if not any(name.endswith(f".{suffix}") for suffix in target_modules):
                continue
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if match is None:
                continue
            layer_id = int(match.group(1))
            named_lora_modules.append((name, module, layer_id))
            layer_ids.append(layer_id)

        if not named_lora_modules:
            return []
        if total_layers is None:
            total_layers = max(layer_ids) + 1

        last_n = getattr(self.args, "orth_lora_last_n_layers", 4)
        last_n = max(1, int(last_n))
        target_layers = set(range(max(0, total_layers - last_n), total_layers))

        targets = []
        for _, module, layer_id in named_lora_modules:
            if layer_id not in target_layers:
                continue
            targets.append(module)
        return targets

    def _pad_description_sequences(self, sequences, dtype=None):
        max_len = max(seq.shape[0] for seq in sequences)
        hidden_size = sequences[0].shape[-1]
        device = sequences[0].device
        if dtype is None:
            dtype = sequences[0].dtype
        padded = torch.zeros((len(sequences), max_len, hidden_size), dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
        for idx, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padded[idx, :seq_len] = seq.to(dtype=dtype)
            mask[idx, :seq_len] = True
        return padded, mask

    def _pad_description_key_masks(self, masks, max_len, device):
        if masks is None:
            return None
        padded = torch.zeros((len(masks), max_len), dtype=torch.bool, device=device)
        for idx, mask in enumerate(masks):
            seq_len = min(mask.shape[0], max_len)
            padded[idx, :seq_len] = mask[:seq_len].to(device=device, dtype=torch.bool)
        return padded

    def _masked_mean_pool(self, hidden_states, mask):
        if hidden_states.shape[1] != mask.shape[1]:
            shared_seq_len = min(hidden_states.shape[1], mask.shape[1])
            padding_side = getattr(self.model.config, "tokenizer_padding_side", "right")
            if padding_side == "left":
                hidden_states = hidden_states[:, -shared_seq_len:]
                mask = mask[:, -shared_seq_len:]
            else:
                hidden_states = hidden_states[:, :shared_seq_len]
                mask = mask[:, :shared_seq_len]
        mask = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _truncate_preserving_targets(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape[0] <= max_length:
            return input_ids, labels

        target_positions = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
        if target_positions.numel() == 0 or target_positions[-1].item() < max_length:
            return input_ids[:max_length], labels[:max_length]

        prefix_len = min(64, max_length - target_positions.numel())
        suffix_len = max_length - prefix_len
        input_ids = torch.cat((input_ids[:prefix_len], input_ids[-suffix_len:]), dim=0)
        labels = torch.cat((labels[:prefix_len], labels[-suffix_len:]), dim=0)
        return input_ids, labels

    def _build_text_only_batch(self, inputs: dict, prefix_len: int) -> dict:
        if "input_ids" not in inputs or "labels" not in inputs or "attention_mask" not in inputs:
            raise ValueError("Building description-utility batch requires input_ids, labels, and attention_mask.")

        max_length = int(getattr(self.args, "model_max_length", 2048) or 2048)
        max_text_len = max(1, max_length - prefix_len)
        device = inputs["input_ids"].device

        sequences = []
        label_sequences = []
        for idx in range(inputs["input_ids"].shape[0]):
            cur_attention = inputs["attention_mask"][idx].bool()
            cur_input_ids = inputs["input_ids"][idx][cur_attention]
            cur_labels = inputs["labels"][idx][cur_attention]

            keep_mask = cur_input_ids.ne(IMAGE_TOKEN_INDEX)
            cur_input_ids = cur_input_ids[keep_mask]
            cur_labels = cur_labels[keep_mask]

            cur_input_ids, cur_labels = self._truncate_preserving_targets(cur_input_ids, cur_labels, max_text_len)
            sequences.append(cur_input_ids)
            label_sequences.append(cur_labels)

        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device=device)
        labels = torch.nn.utils.rnn.pad_sequence(
            label_sequences,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ).to(device=device)
        attention_mask = input_ids.ne(pad_token_id)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def _compute_description_utility_loss(self, model, inputs, description_states):
        base_model = getattr(model, "module", model)
        model_type = getattr(getattr(base_model, "config", None), "model_type", None)
        if model_type is not None and "mpt" in str(model_type):
            current_description_states, current_description_mask = self._pad_description_sequences(description_states)
            description_summary = self._masked_mean_pool(current_description_states.float(), current_description_mask)
            answer_mask = inputs["labels"].ne(IGNORE_INDEX)
            if not torch.any(answer_mask):
                answer_mask = inputs["attention_mask"].bool()
            answer_hidden_states = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                images=inputs.get("images"),
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[self.args.description_hidden_layer]
            answer_summary = self._masked_mean_pool(answer_hidden_states.float(), answer_mask)
            return (1.0 - F.cosine_similarity(answer_summary, description_summary, dim=-1)).mean()

        prefix_states, prefix_mask = self._pad_description_sequences(description_states)
        prefix_embeds = prefix_states.to(dtype=base_model.get_input_embeddings().weight.dtype)
        prefix_attention_mask = prefix_mask
        prefix_len = prefix_embeds.shape[1]

        text_batch = self._build_text_only_batch(inputs, prefix_len=prefix_len)
        token_embeds = base_model.get_input_embeddings()(text_batch["input_ids"])

        inputs_embeds = torch.cat((prefix_embeds, token_embeds), dim=1)
        attention_mask = torch.cat((prefix_attention_mask, text_batch["attention_mask"]), dim=1)
        prefix_labels = torch.full(
            (text_batch["labels"].shape[0], prefix_len),
            IGNORE_INDEX,
            dtype=text_batch["labels"].dtype,
            device=text_batch["labels"].device,
        )
        labels = torch.cat((prefix_labels, text_batch["labels"]), dim=1)

        utility_outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        return utility_outputs.loss

    def _compute_lora_orthogonal_loss(self, model, inputs):
        if getattr(self.args, "orth_lora_weight", 0.0) <= 0:
            return None
        if "description_input_ids" not in inputs:
            return None
        _, _, lora_orth_loss = self._extract_description_states(model, inputs, compute_lora_orth=True)
        return lora_orth_loss

    def _maybe_log_description_losses(
        self,
        total_loss,
        standard_loss,
        description_align_loss,
        description_utility_loss,
        lora_orthogonal_loss,
        valid_token_count,
        base_valid_token_count,
        shared_seq_len,
    ):
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        global_step = int(getattr(self.state, "global_step", 0))
        if global_step % logging_steps != 0:
            return
        if getattr(self, "_last_description_loss_log_step", None) == global_step:
            return
        self._last_description_loss_log_step = global_step

        align_weight = float(getattr(self.args, "description_align_weight", 1.0))
        utility_weight = float(getattr(self.args, "description_utility_weight", 1.0))
        ce_weight = float(getattr(self.args, "standard_ce_weight", 1.0))
        orth_weight = float(getattr(self.args, "orth_lora_weight", 0.0))
        base_valid = base_valid_token_count.detach().float().clamp_min(1.0)
        used_valid = valid_token_count.detach().float()

        logs = {
            "loss/total": total_loss.detach().float().item(),
            "loss/standard_ce": standard_loss.detach().float().item(),
            "loss/description_align": description_align_loss.detach().float().item(),
            "loss/description_utility": description_utility_loss.detach().float().item(),
            "loss_weighted/standard_ce": (standard_loss.detach().float() * ce_weight).item(),
            "loss_weighted/description_align": (description_align_loss.detach().float() * align_weight).item(),
            "loss_weighted/description_utility": (description_utility_loss.detach().float() * utility_weight).item(),
            "description/align_token_fraction": (used_valid / base_valid).item(),
            "description/align_tokens": used_valid.item(),
            "description/base_valid_tokens": base_valid_token_count.detach().float().item(),
            "description/shared_seq_len": float(shared_seq_len),
            "config/description_align_weight": align_weight,
            "config/description_utility_weight": utility_weight,
            "config/standard_ce_weight": ce_weight,
            "config/old_lora_scale": float(getattr(self.args, "old_lora_scale", 1.0)),
            "config/orth_lora_weight": orth_weight,
            "config/orth_lora_last_n_layers": float(getattr(self.args, "orth_lora_last_n_layers", 4)),
        }
        if lora_orthogonal_loss is not None:
            logs["loss/lora_activation_orth"] = lora_orthogonal_loss.detach().float().item()
            logs["loss_weighted/lora_activation_orth"] = (
                lora_orthogonal_loss.detach().float() * orth_weight
            ).item()
        self.log(logs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self.args, "enable_description_cl", False) or "description_input_ids" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        standard_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            images=inputs.get("images"),
            return_dict=True,
            output_hidden_states=False,
            use_cache=False,
        )
        standard_loss = standard_outputs.loss

        if "reference_description_states" not in inputs or "reference_description_mask" not in inputs:
            raise ValueError("Description continual-learning loss requires offline cached reference description states.")

        compute_lora_orth = getattr(self.args, "orth_lora_weight", 0.0) > 0
        if compute_lora_orth:
            description_states, description_key_masks, lora_orthogonal_loss = self._extract_description_states(
                model, inputs, compute_lora_orth=True
            )
        else:
            description_states, description_key_masks = self._extract_description_states(model, inputs)
            lora_orthogonal_loss = None
        description_utility_loss = self._compute_description_utility_loss(model, inputs, description_states)

        current_description_states, current_description_mask = self._pad_description_sequences(
            description_states,
            dtype=inputs["reference_description_states"].dtype,
        )
        reference_description_states = inputs["reference_description_states"].to(current_description_states.device)
        reference_description_mask = inputs["reference_description_mask"].to(current_description_states.device)
        shared_seq_len = min(current_description_states.shape[1], reference_description_states.shape[1])
        current_description_states = current_description_states[:, :shared_seq_len]
        current_description_mask = current_description_mask[:, :shared_seq_len]
        reference_description_states = reference_description_states[:, :shared_seq_len]
        reference_description_mask = reference_description_mask[:, :shared_seq_len]
        valid_mask = current_description_mask & reference_description_mask
        base_valid_token_count = valid_mask.float().sum()
        description_key_mask = self._pad_description_key_masks(
            description_key_masks,
            shared_seq_len,
            current_description_states.device,
        )
        if description_key_mask is not None:
            key_valid_mask = valid_mask & description_key_mask[:, :shared_seq_len]
            has_key_tokens = key_valid_mask.any(dim=1, keepdim=True)
            valid_mask = torch.where(has_key_tokens, key_valid_mask, valid_mask)
        valid_token_count = valid_mask.float().sum()

        diff = (current_description_states.float() - reference_description_states.float()) ** 2
        valid_mask = valid_mask.unsqueeze(-1).float()
        description_align_loss = (diff * valid_mask).sum() / (
            valid_mask.sum().clamp_min(1.0) * current_description_states.shape[-1]
        )

        total_loss = (
            self.args.description_align_weight * description_align_loss
            + self.args.description_utility_weight * description_utility_loss
            + self.args.standard_ce_weight * standard_loss
        )
        if lora_orthogonal_loss is not None:
            total_loss = total_loss + self.args.orth_lora_weight * lora_orthogonal_loss

        self._maybe_log_description_losses(
            total_loss=total_loss,
            standard_loss=standard_loss,
            description_align_loss=description_align_loss,
            description_utility_loss=description_utility_loss,
            lora_orthogonal_loss=lora_orthogonal_loss,
            valid_token_count=valid_token_count,
            base_valid_token_count=base_valid_token_count,
            shared_seq_len=shared_seq_len,
        )

        if return_outputs:
            outputs = {
                "standard_outputs": standard_outputs,
                "standard_loss": standard_loss.detach(),
                "description_utility_loss": description_utility_loss.detach(),
                "description_align_loss": description_align_loss.detach(),
            }
            if lora_orthogonal_loss is not None:
                outputs["lora_orthogonal_loss"] = lora_orthogonal_loss.detach()
            return total_loss, outputs
        return total_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
