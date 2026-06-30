import os
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from contextlib import nullcontext
from contextlib import contextmanager

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
from typing import Any, Dict, List, Optional, Union

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

try:
    from transformers.trainer import smp_forward_backward
except ImportError:
    smp_forward_backward = None

try:
    from apex import amp
except ImportError:
    amp = None


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


class BadBatchError(RuntimeError):
    pass


_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_INDEX_PATTERN = re.compile(r"(?:^|\.)lora[AB]\.(\d+)(?:\.|$)")


@contextmanager
def _temporary_attr(obj, name: str, value):
    has_attr = hasattr(obj, name)
    prev_value = getattr(obj, name, None)
    try:
        setattr(obj, name, value)
        yield
    finally:
        if has_attr:
            try:
                setattr(obj, name, prev_value)
            except Exception:
                pass
        else:
            try:
                delattr(obj, name)
            except Exception:
                pass


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
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

    def _sync_skip_flag(self, local_skip: bool, device: torch.device) -> bool:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local_skip
        skip_tensor = torch.tensor(1 if local_skip else 0, device=device, dtype=torch.int32)
        torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.MAX)
        return bool(skip_tensor.item())

    def _summarize_batch(self, inputs: Dict[str, Any]) -> str:
        parts = []
        attention_mask = inputs.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            lengths = attention_mask.long().sum(dim=1).tolist()
            parts.append(f"text_lens={lengths}")
        labels = inputs.get("labels")
        if torch.is_tensor(labels):
            valid_counts = labels.ne(IGNORE_INDEX).long().sum(dim=1).tolist()
            parts.append(f"valid_label_counts={valid_counts}")
        description_attention_mask = inputs.get("description_attention_mask")
        if torch.is_tensor(description_attention_mask) and description_attention_mask.ndim == 2:
            desc_lengths = description_attention_mask.long().sum(dim=1).tolist()
            parts.append(f"description_lens={desc_lengths}")
        if "images" in inputs:
            images = inputs["images"]
            if torch.is_tensor(images):
                parts.append(f"images_shape={tuple(images.shape)}")
            elif isinstance(images, list):
                parts.append(f"images_list_len={len(images)}")
        cache_keys = inputs.get("description_cache_keys")
        if isinstance(cache_keys, list) and cache_keys:
            parts.append(f"cache_keys={cache_keys[:2]}")
        return " ".join(parts)

    def _record_bad_batch(self, message: str):
        total = int(getattr(self, "_bad_batch_total", 0)) + 1
        consecutive = int(getattr(self, "_bad_batch_consecutive", 0)) + 1
        self._bad_batch_total = total
        self._bad_batch_consecutive = consecutive
        if self.args.local_rank in (-1, 0):
            step = int(getattr(self.state, "global_step", 0))
            print(
                f"[bad-batch][step={step}][total_skipped={total}][consecutive={consecutive}] {message}",
                flush=True,
            )
        max_consecutive = int(getattr(self.args, "max_consecutive_bad_batches", 20) or 20)
        if consecutive > max_consecutive:
            raise RuntimeError(
                f"Skipped {consecutive} consecutive bad batches, exceeding max_consecutive_bad_batches={max_consecutive}."
            )

    def _reset_bad_batch_streak(self):
        self._bad_batch_consecutive = 0

    def _ensure_finite_loss(self, name: str, value: torch.Tensor, inputs: Dict[str, Any]):
        if torch.is_tensor(value) and not torch.isfinite(value.detach()).all():
            raise BadBatchError(f"{name} became non-finite. {self._summarize_batch(inputs)}")

    def _resolve_num_hidden_layers(self, model) -> Optional[int]:
        wrapped = getattr(model, "module", model)
        for candidate in (
            getattr(getattr(wrapped, "config", None), "num_hidden_layers", None),
            getattr(getattr(wrapped, "config", None), "n_layer", None),
            getattr(getattr(wrapped, "config", None), "num_layers", None),
        ):
            if candidate is None:
                continue
            try:
                candidate = int(candidate)
            except (TypeError, ValueError):
                continue
            if candidate > 0:
                return candidate

        max_layer_idx = -1
        for name, _ in wrapped.named_parameters():
            match = _LAYER_INDEX_PATTERN.search(name)
            if match is None:
                continue
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
        if max_layer_idx >= 0:
            return max_layer_idx + 1
        return None

    def _should_mask_aux_grad(self, param_name: str, target_expert: int, layer_cutoff: int) -> bool:
        expert_match = _EXPERT_INDEX_PATTERN.search(param_name)
        if expert_match is None:
            return False
        if int(expert_match.group(1)) != int(target_expert):
            return False
        layer_match = _LAYER_INDEX_PATTERN.search(param_name)
        if layer_match is None:
            return False
        return int(layer_match.group(1)) >= int(layer_cutoff)

    @contextmanager
    def _temporary_aux_grad_mask(self, model):
        wrapped = getattr(model, "module", model)
        if not getattr(self.args, "enable_layerwise_aux_loss", False):
            yield
            return

        cur_task = getattr(wrapped, "cur_task", None)
        if cur_task is None:
            yield
            return

        exclude_last_n_layers = int(getattr(self.args, "aux_exclude_last_n_layers", 0) or 0)
        if exclude_last_n_layers <= 0:
            yield
            return

        num_hidden_layers = self._resolve_num_hidden_layers(model)
        if num_hidden_layers is None:
            yield
            return

        layer_cutoff = max(0, num_hidden_layers - exclude_last_n_layers)
        hooks = []

        def _zero_aux_grad(grad):
            if grad is None:
                return None
            return torch.zeros_like(grad)

        try:
            for name, param in wrapped.named_parameters():
                if not param.requires_grad:
                    continue
                if self._should_mask_aux_grad(name, int(cur_task), layer_cutoff):
                    hooks.append(param.register_hook(_zero_aux_grad))
            yield
        finally:
            for hook in hooks:
                try:
                    hook.remove()
                except Exception:
                    pass

    def _cap_loss_value(self, loss: torch.Tensor, cap: float):
        if cap is None or float(cap) <= 0:
            return loss
        return torch.clamp(loss, max=float(cap))

    def _compute_description_loss_bundle(self, model, inputs):
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
        self._ensure_finite_loss("standard_loss", standard_loss, inputs)

        if "reference_description_states" not in inputs or "reference_description_mask" not in inputs:
            raise ValueError("Description continual-learning loss requires offline cached backbone reference description states.")

        current_states, current_mask = self._extract_description_states(
            model,
            inputs,
            disable_adapter=False,
            disable_anchor_update=True,
        )
        reference_states = inputs["reference_description_states"].to(current_states.device)
        reference_mask = inputs["reference_description_mask"].to(current_states.device)
        shared_seq_len = min(current_states.shape[1], reference_states.shape[1])
        current_states = current_states[:, :shared_seq_len]
        current_mask = current_mask[:, :shared_seq_len]
        reference_states = reference_states[:, :shared_seq_len]
        reference_mask = reference_mask[:, :shared_seq_len]

        valid_mask = current_mask & reference_mask
        diff = (current_states.float() - reference_states.float()) ** 2
        mask_f = valid_mask.unsqueeze(-1).float()
        description_align_loss_raw = (diff * mask_f).sum() / (mask_f.sum().clamp_min(1.0) * current_states.shape[-1])
        self._ensure_finite_loss("description_align_loss", description_align_loss_raw, inputs)

        description_align_loss = self._cap_loss_value(description_align_loss_raw, self.args.description_align_loss_cap)
        utility_weight = float(getattr(self.args, "description_utility_weight", 0.0) or 0.0)
        if utility_weight != 0.0:
            description_utility_loss_raw = self._compute_description_utility_loss(model, inputs, reference_states, reference_mask)
            self._ensure_finite_loss("description_utility_loss", description_utility_loss_raw, inputs)
            description_utility_loss = self._cap_loss_value(description_utility_loss_raw, self.args.description_utility_loss_cap)
        else:
            description_utility_loss_raw = None
            description_utility_loss = None
        total_loss = (
            self.args.standard_ce_weight * standard_loss
            + self.args.description_align_weight * description_align_loss
            + (utility_weight * description_utility_loss if description_utility_loss is not None else 0.0)
        )
        self._ensure_finite_loss("total_loss", total_loss, inputs)

        bundle = {
            "standard_outputs": standard_outputs,
            "standard_loss": standard_loss,
            "description_align_loss_raw": description_align_loss_raw,
            "description_align_loss": description_align_loss,
            "description_utility_loss_raw": description_utility_loss_raw,
            "description_utility_loss": description_utility_loss,
            "total_loss": total_loss,
        }
        return bundle

    def _log_loss_bundle(self, bundle: Dict[str, torch.Tensor], cur_task):
        if self.args.local_rank not in (-1, 0) or getattr(self, "state", None) is None:
            return

        def _to_float(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if torch.is_tensor(value):
                return float(value.detach().float().mean().cpu())
            return None

        def _finite_flag(value):
            if value is None:
                return "none"
            if torch.is_tensor(value):
                finite = torch.isfinite(value.detach()).all().item()
                return "finite" if finite else "nonfinite"
            if isinstance(value, (int, float)):
                return "finite" if float(value) == float(value) and abs(float(value)) != float("inf") else "nonfinite"
            return "unknown"

        step = int(getattr(self.state, "global_step", 0))
        print(
            f"[loss][step={step}][cur_task={cur_task}] "
            f"standard_ce={_to_float(bundle['standard_loss'])}({_finite_flag(bundle['standard_loss'])}) "
            f"description_align_raw={_to_float(bundle['description_align_loss_raw'])}({_finite_flag(bundle['description_align_loss_raw'])}) "
            f"description_align={_to_float(bundle['description_align_loss'])}({_finite_flag(bundle['description_align_loss'])}) "
            f"description_utility_raw={_to_float(bundle['description_utility_loss_raw'])}({_finite_flag(bundle['description_utility_loss_raw'])}) "
            f"description_utility={_to_float(bundle['description_utility_loss'])}({_finite_flag(bundle['description_utility_loss'])}) "
            f"total={_to_float(bundle['total_loss'])}({_finite_flag(bundle['total_loss'])})",
            flush=True,
        )

    def _backward_loss(self, loss: torch.Tensor):
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        local_skip = False
        local_reason = ""
        loss = None
        bundle = None
        layerwise_aux_enabled = bool(getattr(self.args, "enable_layerwise_aux_loss", False))
        force_bad_step = int(getattr(self.args, "debug_force_bad_batch_step", -1) or -1)
        current_step = int(getattr(self.state, "global_step", 0))
        if (
            force_bad_step >= 0
            and current_step == force_bad_step
            and not getattr(self, "_debug_forced_bad_batch_done", False)
        ):
            self._debug_forced_bad_batch_done = True
            local_skip = True
            local_reason = f"BadBatchError: debug_force_bad_batch_step={force_bad_step}"

        if not local_skip:
            try:
                with self.compute_loss_context_manager():
                    if (
                        layerwise_aux_enabled
                        and "description_input_ids" in inputs
                        and getattr(getattr(model, "module", model), "cur_task", None) is not None
                        and int(getattr(getattr(model, "module", model), "cur_task", 0)) > 0
                    ):
                        bundle = self._compute_description_loss_bundle(model, inputs)
                        loss = bundle["total_loss"]
                    else:
                        loss = self.compute_loss(model, inputs)
                if torch.is_tensor(loss) and not torch.isfinite(loss.detach()).all():
                    raise BadBatchError(f"total_loss became non-finite. {self._summarize_batch(inputs)}")
            except Exception as exc:
                if not getattr(self.args, "skip_bad_batches", True):
                    raise
                local_skip = True
                local_reason = f"{type(exc).__name__}: {exc}"

        sync_device = self.args.device
        if torch.is_tensor(loss):
            sync_device = loss.device
        elif "labels" in inputs and torch.is_tensor(inputs["labels"]):
            sync_device = inputs["labels"].device
        should_skip = self._sync_skip_flag(local_skip, sync_device)

        if should_skip:
            if local_skip:
                self._record_bad_batch(local_reason)
            elif self.args.local_rank in (-1, 0):
                self._record_bad_batch("Skipped because another rank reported a bad batch.")
            return torch.zeros((), device=sync_device)

        self._reset_bad_batch_streak()

        if self.args.n_gpu > 1 and torch.is_tensor(loss):
            loss = loss.mean()

        if not layerwise_aux_enabled or bundle is None:
            self._backward_loss(loss)
            return loss.detach() / self.args.gradient_accumulation_steps

        self._log_loss_bundle(bundle, cur_task=getattr(getattr(model, "module", model), "cur_task", None))

        self._backward_loss(bundle["standard_loss"] * self.args.standard_ce_weight)

        aux_backprops = []
        if float(self.args.description_align_weight) != 0.0:
            aux_backprops.append(bundle["description_align_loss"] * self.args.description_align_weight)
        if bundle.get("description_utility_loss") is not None and float(self.args.description_utility_weight) != 0.0:
            aux_backprops.append(bundle["description_utility_loss"] * self.args.description_utility_weight)

        if aux_backprops:
            with self._temporary_aux_grad_mask(model):
                for aux_loss in aux_backprops:
                    self._backward_loss(aux_loss)

        return loss.detach() / self.args.gradient_accumulation_steps

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

    def _masked_mean_pool(self, hidden_states, mask):
        mask = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _truncate_preserving_targets(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_length: int,
    ):
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

    @contextmanager
    def _temporary_adapter(self, model, adapter_name: Optional[str]):
        if not adapter_name:
            yield
            return
        wrapped = getattr(model, "module", model)
        if not hasattr(wrapped, "set_adapter"):
            yield
            return
        prev_adapter = getattr(wrapped, "active_adapter", None)
        try:
            wrapped.set_adapter(adapter_name)
            yield
        finally:
            if isinstance(prev_adapter, str):
                try:
                    wrapped.set_adapter(prev_adapter)
                except Exception:
                    pass

    @contextmanager
    def _temporary_cur_task(self, model, cur_task: Optional[int]):
        if cur_task is None:
            yield
            return
        wrapped = getattr(model, "module", model)
        if not hasattr(wrapped, "cur_task"):
            yield
            return
        prev_task = getattr(wrapped, "cur_task", None)
        try:
            wrapped.cur_task = int(cur_task)
            yield
        finally:
            if prev_task is not None:
                try:
                    wrapped.cur_task = prev_task
                except Exception:
                    pass

    @contextmanager
    def _temporary_anchor_update(self, model, enabled: bool):
        wrapped = getattr(model, "module", model)
        disable_updates = not bool(enabled)
        with _temporary_attr(wrapped, "disable_anchor_update", disable_updates):
            yield

    def _extract_description_states(
        self,
        model,
        inputs,
        disable_adapter: bool,
        dtype=None,
        adapter_name: Optional[str] = None,
        cur_task: Optional[int] = None,
        disable_anchor_update: bool = False,
    ):
        wrapped = getattr(model, "module", model)
        ctx = nullcontext()
        if disable_adapter and hasattr(wrapped, "disable_adapter"):
            ctx = wrapped.disable_adapter()

        with (
            ctx,
            self._temporary_adapter(model, adapter_name),
            self._temporary_cur_task(model, cur_task),
            self._temporary_anchor_update(model, enabled=not disable_anchor_update),
        ):
            description_outputs = model(
                input_ids=inputs["description_input_ids"],
                attention_mask=inputs["description_attention_mask"],
                images=inputs.get("images"),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = description_outputs.hidden_states[self.args.description_hidden_layer]
        sequences = []
        lengths = inputs["description_attention_mask"].long().sum(dim=1).tolist()
        for batch_idx, cur_len in enumerate(lengths):
            start_idx = max(0, cur_len - self.args.description_max_tokens)
            sequences.append(hidden_states[batch_idx, start_idx:cur_len])
        return self._pad_description_sequences(sequences, dtype=dtype)

    def _compute_description_utility_loss(self, model, inputs, prefix_states, prefix_mask):
        base_model = getattr(model, "module", model)
        model_type = getattr(getattr(base_model, "config", None), "model_type", None)
        if model_type is not None and "mpt" in str(model_type):
            description_summary = self._masked_mean_pool(prefix_states.float(), prefix_mask)
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

        embed_dtype = base_model.get_input_embeddings().weight.dtype
        prefix_embeds = prefix_states.to(dtype=embed_dtype)
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        wrapped = getattr(model, "module", model)
        cur_task = getattr(wrapped, "cur_task", None)
        if (
            not getattr(self.args, "enable_description_cl", False)
            or "description_input_ids" not in inputs
            or (cur_task is not None and int(cur_task) <= 0)
        ):
            try:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
        bundle = self._compute_description_loss_bundle(model, inputs)
        self._log_loss_bundle(bundle, cur_task=cur_task)
        if return_outputs:
            outputs = {
                "standard_outputs": bundle["standard_outputs"],
                "standard_loss": bundle["standard_loss"].detach(),
                "description_align_loss_raw": bundle["description_align_loss_raw"].detach(),
                "description_align_loss": bundle["description_align_loss"].detach(),
                "description_utility_loss_raw": bundle["description_utility_loss_raw"].detach(),
                "description_utility_loss": bundle["description_utility_loss"].detach(),
            }
            return bundle["total_loss"], outputs
        return bundle["total_loss"]

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

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
