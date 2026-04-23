import os
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

from llava.constants import IGNORE_INDEX


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

    def _extract_description_states(self, model, inputs):
        description_outputs = model(
            input_ids=inputs["description_input_ids"],
            attention_mask=inputs["description_attention_mask"],
            images=inputs.get("images"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = description_outputs.hidden_states[self.args.description_hidden_layer]
        description_sequences = []
        lengths = inputs["description_attention_mask"].long().sum(dim=1).tolist()
        for batch_idx, cur_len in enumerate(lengths):
            start_idx = max(0, cur_len - self.args.description_max_tokens)
            description_sequences.append(hidden_states[batch_idx, start_idx:cur_len])
        return description_sequences

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

    def _compute_description_utility_loss(self, standard_outputs, inputs, description_states):
        current_description_states, current_description_mask = self._pad_description_sequences(description_states)
        description_summary = self._masked_mean_pool(current_description_states.float(), current_description_mask)

        answer_hidden_states = standard_outputs.hidden_states[self.args.description_hidden_layer]
        prepared_labels = getattr(standard_outputs, "prepared_labels", None)
        prepared_attention_mask = getattr(standard_outputs, "prepared_attention_mask", None)
        if prepared_labels is not None:
            answer_mask = prepared_labels.ne(IGNORE_INDEX)
        else:
            answer_mask = inputs["labels"].ne(IGNORE_INDEX)
        if not torch.any(answer_mask):
            if prepared_attention_mask is not None:
                answer_mask = prepared_attention_mask.bool()
            else:
                answer_mask = inputs["attention_mask"].bool()
        answer_summary = self._masked_mean_pool(answer_hidden_states.float(), answer_mask)

        return (1.0 - F.cosine_similarity(answer_summary, description_summary, dim=-1)).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self.args, "enable_description_cl", False) or "description_input_ids" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        standard_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            images=inputs.get("images"),
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        standard_loss = standard_outputs.loss

        if "reference_description_states" not in inputs or "reference_description_mask" not in inputs:
            raise ValueError("Description continual-learning loss requires offline cached reference description states.")

        description_states = self._extract_description_states(model, inputs)
        description_utility_loss = self._compute_description_utility_loss(standard_outputs, inputs, description_states)

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

        if return_outputs:
            outputs = {
                "standard_outputs": standard_outputs,
                "standard_loss": standard_loss.detach(),
                "description_utility_loss": description_utility_loss.detach(),
                "description_align_loss": description_align_loss.detach(),
            }
            return total_loss, outputs
        return total_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
