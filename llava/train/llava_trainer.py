from llava.mm_utils import IMAGE_TOKEN_INDEX
from llava.eval_metrics import MAP, MACRO_F1, stem_list, PorterStemmer

from typing import List, Optional
import os
import torch
import torch.nn as nn
from torch.distributed import all_reduce, ReduceOp
from deepspeed import DeepSpeedEngine
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    EvalLoopOutput,
    TRAINER_STATE_NAME
)
from typing import List, Optional
    
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stride map for Progressive Modality Masking
        self.vcmp_map = torch.ones(self.train_dataset.__len__(), dtype=torch.long, device=self.args.device) * self.args.init_factor
        self.tcmp_map = torch.ones(self.train_dataset.__len__(), dtype=torch.long, device=self.args.device) * self.args.init_factor
        if self.args.use_aimkp:
            # Initialize gradient storage and other variables for gradient computation and accumulation
            self.grads = {}
            self.lm_keys = []
            self.mm_keys = []
            self.model_keys = []
            self.param_list = []
            for name, param in self.model.get_model().named_parameters():
                if param.requires_grad is False:
                    continue
                self.model_keys.append(name)
                self.param_list.append(param)
                self.grads[name] = torch.zeros_like(param.data).float()
                if "mm_projector" in name:
                    self.mm_keys.append(name)
                else:
                    self.lm_keys.append(name)
            self.total_batched_samples = 0
            self.tokens_count = 0

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
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad )
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
                    }
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

    # Patched forward function for AimKP
    def patched_forward(
        self,
        model,
        input_ids = None,
        attention_mask = None,
        position_ids= None,
        past_key_values = None,
        inputs_embeds = None,
        labels= None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        images = None,
        image_sizes = None,
        text_ids = None,
        text_attention_mask = None,
        **kwargs
    ):
        idx = kwargs.pop("id", None)
        outputs = []
        assert input_ids.shape[0] == 1, "Only supports batch size of 1 for cmp mode."
        t_factor = torch.abs(self.tcmp_map[idx]).item()
        v_factor = torch.abs(self.vcmp_map[idx]).item()
        b_size = 1 + (int(t_factor > 1)) + (int(v_factor > 1))
        input_ids = input_ids.repeat(b_size, 1) # bsz * seq_len
        labels = labels.repeat(b_size, 1) # bsz * seq_len
        images = images.repeat(b_size, 1, 1, 1).to(model.dtype) # bsz * c * h * w
        text_attention_mask = text_attention_mask.repeat(b_size, 1) # bsz * seq_len
        # Pre-processing inputs for training
        attention_mask = attention_mask.repeat(b_size,  1) # bsz * seq_len
        (
            _,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes
        )
        # Compute tokens' position of each modality
        len_image = 576
        h = w = 24
        len_text = text_attention_mask[0].sum().item()
        image_begin = torch.nonzero(input_ids[0] == IMAGE_TOKEN_INDEX, as_tuple=True)[0][0]
        image_end = image_begin + len_image - 1
        text_begin = image_end + 1
        text_end = text_begin + len_text - 1
        _index = 1

        # Create attention mask for each modality
        if t_factor > 1:
            text_mask = torch.zeros(len_text, device=model.device, dtype=torch.bool)
            if t_factor <= len_text:
                text_mask[t_factor - 1::t_factor] = True
            else:
                text_mask[-1] = True
            attention_mask[_index][text_begin:text_end + 1] = text_mask
            _index += 1
        if v_factor > 1:
            image_mask = torch.zeros(len_image, device=model.device, dtype=torch.bool)
            if v_factor <= h or v_factor <= w:
                image_mask = image_mask.view(h, w)
                image_mask[v_factor - 1::v_factor, v_factor - 1::v_factor] = True
                image_mask = image_mask.reshape(-1)
            else:
                image_mask[-1] = True
            attention_mask[_index][image_begin:image_end + 1] = image_mask

        # Forward of normal, text-masked, image-masked inputs
        outputs = []
        for _inputs_embeds, _attention_mask, _labels in zip(inputs_embeds, attention_mask, labels):
            _inputs_embeds = _inputs_embeds.unsqueeze(0) 
            _attention_mask = _attention_mask.unsqueeze(0)
            _labels = _labels.unsqueeze(0)
            outputs.append(model(
                input_ids=None,
                attention_mask=_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=_inputs_embeds,
                labels=_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            ))
        loss = [output["loss"] for output in outputs] if return_dict else [output[0] for output in outputs]
        return loss

    # For syncing stride map in distributed training
    def sync_cmp_map(self):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            world_size = torch.distributed.get_world_size()
            gathered_list = [torch.zeros_like(self.vcmp_map) for _ in range(world_size)]

            torch.distributed.all_gather(gathered_list, self.vcmp_map)
            self.vcmp_map = torch.stack(gathered_list, dim=0)
            min_vals = self.vcmp_map.min(dim=0)[0]
            has_negative = (self.vcmp_map < 0).any(dim=0)
            max_vals = self.vcmp_map.max(dim=0)[0]
            self.vcmp_map = torch.where(has_negative, min_vals, max_vals)

            torch.distributed.all_gather(gathered_list, self.tcmp_map)
            self.tcmp_map = torch.stack(gathered_list, dim=0)
            min_vals = self.tcmp_map.min(dim=0)[0]
            has_negative = (self.tcmp_map < 0).any(dim=0)
            max_vals = self.tcmp_map.max(dim=0)[0]
            self.tcmp_map = torch.where(has_negative, min_vals, max_vals)

        return

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # This is for standard training step
        if not self.args.use_aimkp:
            inputs.pop('id', None)
            inputs.pop('text_ids', None)
            inputs.pop('text_attention_mask', None)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps
        
        if torch.distributed.is_initialized():
            model = model.module
        loss = self.patched_forward(model=model, **inputs)
        data_id = inputs.pop('id', 0)

        # Store the number of label tokens to weight the losses of a batch, and gpus accordingly.
        labels_tokens_num = torch.sum(inputs['labels'] != -100)
        self.tokens_count += labels_tokens_num
        grads_list = []

        # Per-sample grad computation, we apply multi-backward, for efficiency you can try GradSampleModule from Opacus
        for idx, l in enumerate(loss):
            grads_list.append({})
            if idx < len(loss) - 1:
                l.backward(retain_graph=True)
            else:
                l.backward()
            for name, param in self.model.get_model().named_parameters():
                if param.grad is not None:
                    grads_list[idx][name] = param.grad.data.clone()
            model.zero_grad()

        # Thresholds for gradient cosine similarity
        tcmp_thres = self.args.tcmp_thres
        vcmp_thres = self.args.vcmp_thres
        min_thres = []
        if torch.abs(self.tcmp_map[data_id]) > 1:
            min_thres.append(tcmp_thres)
        if torch.abs(self.vcmp_map[data_id]) > 1:
            min_thres.append(vcmp_thres)

        # Gradient-Based Filtering: filter gradients based on cosine similarity equal to filtering the losses
        cosine_scores = []
        remove_id = []
        grads_concat_s = torch.cat([grads_list[0][name].view(-1) for name in self.model_keys])
        for i in range(1, len(grads_list)):
            cur_grads = grads_list[i]
            grads_concat_cur = torch.cat([cur_grads[name].view(-1) for name in self.model_keys])
            grads_cos = torch.nn.functional.cosine_similarity(grads_concat_s, grads_concat_cur, dim=0)
            cosine_scores.append(grads_cos)
            if grads_cos < min_thres[i - 1]:
                remove_id.append(i)
        if len(remove_id) > 0:
            for i in sorted(remove_id, reverse=True):
                grads_list.pop(i)

        # Aggregate gradients of the remaining samples
        for name in self.grads.keys():
            g = []
            for i in range(len(grads_list)):
                if name in grads_list[i].keys():
                    g.append(grads_list[i][name].float())
            # We simply average the loss by averaging the gradients
            self.grads[name] += torch.stack(g, dim=0).mean(dim=0) * labels_tokens_num 


        # Get stride γ of this sample
        t_factor = self.tcmp_map[data_id]
        v_factor = self.vcmp_map[data_id]

        # Training information
        # with open(os.path.join(self.args.output_dir,'train_details.txt'), 'a') as f:
        #     f.write(f'id:{data_id.item()}, t_factor:{t_factor.item()}, v_factor:{v_factor.item()}, cosine_scores:{[c.item() for c in cosine_scores]}, loss:{[l.item() for l in loss]}\n')

        # Update stride γ according to the gradient cosine similarity
        f_index = 0
        if t_factor == 1:
            self.tcmp_map[data_id] = t_factor * 2
        elif t_factor > 1:
            if cosine_scores[f_index] > tcmp_thres:
                self.tcmp_map[data_id] = t_factor * 2
            else:
                self.tcmp_map[data_id] = ( -t_factor // 2) if (- t_factor // 2) < -2 else -2
            f_index += 1

        if v_factor == 1:
            self.vcmp_map[data_id] = v_factor * 2
        elif v_factor > 1:
            if cosine_scores[f_index] > vcmp_thres:
                self.vcmp_map[data_id] = v_factor * 2
            else:
                self.vcmp_map[data_id] = ( -v_factor // 2) if (- v_factor // 2) < -2 else -2
            f_index += 1

        # For gradient accumulation as this version of transformer trainer has bug in gradient accumulation
        self.total_batched_samples = self.total_batched_samples + 1
        if (
            self.total_batched_samples % self.args.gradient_accumulation_steps == 0
        ):
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                tokens_count_tensor = torch.tensor(self.tokens_count, device=self.args.device)
                all_reduce(tokens_count_tensor, op=ReduceOp.SUM)
                self.tokens_count = tokens_count_tensor.item()
            for name, param in self.model.get_model().named_parameters():
                if name in self.grads.keys():
                    param.grad = (self.grads[name] / self.tokens_count).to(param.dtype)
                    self.grads[name].zero_() 
            self.tokens_count = 0
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                for param in self.param_list:
                    all_reduce(param.grad, op=ReduceOp.SUM)

        loss = loss[0]
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            all_reduce(loss, op=ReduceOp.AVG)
        return loss.detach() / self.args.gradient_accumulation_steps

    def _save_checkpoint(self, model, trial, metrics=None):
        if metrics is not None:
            metric = metrics[self.args.metric_for_best_model]
            if not hasattr(self, "best_metric"):
                self.best_metric = 0
                self.best_step = self.state.global_step
            if self.best_metric <= metric:
                self.best_metric = metric
                self.best_step = self.state.global_step
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = "best-checkpoint"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
                    f.write(f"step: {self.state.global_step}\n")
                    f.write(f"metric: {metrics}\n")
                state_dict = get_peft_state_maybe_zero_3(
                    self.model.named_parameters(), self.args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    self.model.named_parameters()
                )
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    print(f"save models to {output_dir} ")
                    self.model.config.save_pretrained(output_dir)
                    self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
                # rotate checkpoints
                if self.args.should_save:
                    self._rotate_checkpoints(use_mtime=True, output_dir=self.args.output_dir)
                
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
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                os.makedirs(output_dir, exist_ok=True)
                if metrics is not None:
                    with open(os.path.join(self.args.output_dir, "metrics.txt"), 'a') as f:
                        f.write(f"step: {self.state.global_step}\n")
                        f.write(f"metric: {metrics}\n")
                        f.write(f"best_step: {self.best_step}\n")
                        f.write(f"best_metric: {self.best_metric}\n")
                state_dict = get_peft_state_maybe_zero_3(
                    self.model.named_parameters(), self.args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    self.model.named_parameters()
                )
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    print(f"save models to {output_dir} ")
                    self.model.config.save_pretrained(output_dir)
                    self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            else:
                super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        if len(inputs["input_ids"]) != 1:
            raise ValueError("Only supports batch size of 1.")
        labels = inputs.pop("labels")
        image_tensor = inputs.pop("images", None)
        image_size = inputs.pop("image_size", None)
        input_ids = inputs.pop("input_ids")
        label_mask = labels == -100
        input_ids = input_ids[label_mask].unsqueeze(0)  

        labels = labels[labels != -100]
        max_new_tokens = min(40, labels.shape[-1] * 2)
        gen_kwargs = {
            "images": image_tensor.to(model.dtype) if image_tensor is not None else None,
            "image_sizes": [image_size] if image_size is not None else None,
            "do_sample": False,
            "temperature": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True
        }
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                **gen_kwargs
            )

        pred_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        labels = self.tokenizer.decode(labels, skip_special_tokens=True)
        labels = labels.lower().split(",")
        labels = [l.strip() for l in labels]
        preds = pred_text.lower().split(",")
        preds = [p.strip() for p in preds]

        stemmer = PorterStemmer()
        labels = stem_list(labels,stemmer)
        preds = stem_list(preds,stemmer)

        self.map_5.compute_MAP(labels, preds)
        self.f1_1.compute_F1(labels, preds)
        self.f1_3.compute_F1(labels, preds)
        return (None, None, None)  
    
        
    def evaluation_loop(self, *args, **kwargs):
        self.map_5 = MAP(5)
        self.f1_1 = MACRO_F1(1)
        self.f1_3 = MACRO_F1(3)
        metrics = {}

        output = super().evaluation_loop(*args, **kwargs)
        
        map_5_score = torch.tensor(self.map_5.MAP_score, device=self.model.device)
        f1_1_score = torch.tensor(self.f1_1.F1_score, device=self.model.device)
        f1_3_score = torch.tensor(self.f1_3.F1_score, device=self.model.device)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            all_reduce(map_5_score, op=ReduceOp.AVG)
            all_reduce(f1_1_score, op=ReduceOp.AVG)
            all_reduce(f1_3_score, op=ReduceOp.AVG)
            metrics = {
                "f1@1": f1_1_score.item(),
                "f1@3": f1_3_score.item(),
                "map@5": map_5_score.item(),
                "sum":map_5_score.item() + f1_1_score.item() + f1_3_score.item()
            }
            return EvalLoopOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=metrics, num_samples=output.num_samples)
        map_5_score = self.accelerator.gather(map_5_score)
        f1_1_score = self.accelerator.gather(f1_1_score)
        f1_3_score = self.accelerator.gather(f1_3_score)
        metrics = {
            "f1@1": f1_1_score.mean().item(),
            "f1@3": f1_3_score.mean().item(),
            "map@5": map_5_score.mean().item(),
            "sum":map_5_score.mean().item() + f1_1_score.mean().item() + f1_3_score.mean().item()
        }
        
        return EvalLoopOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=metrics, num_samples=output.num_samples)

