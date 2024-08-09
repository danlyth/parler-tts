#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This version of the training script expects pre-prepared DAC tokens and
# can be conditioned on reference audio or natural languge descriptions.

"""Train Parler-TTS using 🤗 Accelerate"""

import logging
import os
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
import warnings

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, set_seed
from accelerate.utils.memory import release_memory
from datasets import IterableDataset
from huggingface_hub import Repository, create_repo
from multiprocess import set_start_method
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.utils import send_example_telemetry

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
)
from training.arguments import DataTrainingArguments, ModelArguments, ParlerTTSTrainingArguments
from training.data_local import DataCollator, DatasetLocal
from training.data_mds import DataLoaderMDS
from training.eval import compute_metrics
from training.utils import get_last_checkpoint, log_metric, log_pred, rotate_checkpoints


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_parler_tts", model_args, data_args)

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
    else:
        mixed_precision = "no"

    if data_args.pad_to_max_length and (
        data_args.max_audio_token_length is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "`pad_to_max_length` is `True` but one of the following parameters has not been set: `max_audio_token_length`, `max_prompt_token_length`, `max_description_token_length`"
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    # Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Accelerator preparation
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=60))]
    if training_args.torch_compile:
        # TODO(YL): add more compile modes?
        kwargs_handlers.append(TorchDynamoPlugin(backend="inductor", mode="default"))  # reduce-overhead

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        # dispatch_batches=False,  # TODO (Dan) testing this as our batches are not all the same length # NOTE commenteing this out as a test
        # split_batches=True,  # NOTE testing this
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        init_kwargs={"wandb": {"save_code": True}, "name": data_args.wandb_run_name},
        config={**vars(training_args), **vars(data_args), **vars(model_args), "name": data_args.wandb_run_name},
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    if not accelerator.is_main_process:
        warnings.filterwarnings("ignore")  # TODO, perhaps duplicating here

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's instantiate the tokenizers and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    sample_rate = model_args.discrete_audio_feature_sample_rate

    # load prompt tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",  # prompt has to be padded on the left bc it's preprend to codebooks hidden states
    )

    if model_args.use_fast_tokenizer:
        logger.warning(
            "Disabling fast tokenizer warning: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3231-L3235"
        )
        prompt_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # load audio reference encoder
    audio_ref_encoder = AutoModel.from_pretrained(
        model_args.audio_ref_encoder_name, output_hidden_states=True, output_attentions=True
    )
    audio_ref_encoder.to(training_args.device)
    audio_ref_encoder.eval()
    if model_args.audio_ref_encoder_hidden_layer is not None:
        logger.info(f"Using hidden layer {model_args.audio_ref_encoder_hidden_layer}")

    # 3. Next, let's load the config.
    config = ParlerTTSConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # update pad token id and decoder_start_token_id
    config.update(
        {
            "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else config.decoder_start_token_id,
        }
    )

    # create model
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze Encoders
    model.freeze_encoders(model_args.freeze_text_encoder)

    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
    audio_encoder_eos_token_id = config.decoder.eos_token_id

    logger.info("Loading datasets...")
    # Instantiate custom data collator
    data_collator = DataCollator(
        prompt_tokenizer=prompt_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        # description_max_length=data_args.max_description_token_length,
        audio_max_length=data_args.max_audio_token_length,
        audio_ref_max_length=data_args.max_audio_ref_length,
    )

    if training_args.do_train:
        if data_args.use_mds:
            train_dataloader = DataLoaderMDS(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                manifest_path=data_args.mds_train_manifest_path,
                batch_size=training_args.per_device_train_batch_size,
                prompt_tokenizer=prompt_tokenizer,
                audio_encoder_bos_token_id=audio_encoder_bos_token_id,
                audio_encoder_eos_token_id=audio_encoder_eos_token_id,
                shuffle=True,
                epoch_size=data_args.max_train_samples,
                collator=data_collator,
                drop_last=True,
            )

        else:
            train_dataset_local = DatasetLocal(
                root_audio_dir=data_args.finetune_audio_dir,
                root_dac_dir=data_args.finetune_code_dir,
                metadata_path=data_args.finetune_train_metadata_path,
                prompt_tokenizer=prompt_tokenizer,
                audio_sr=model_args.audio_ref_encoder_sr,
                audio_ref_len=model_args.audio_ref_len,
                audio_ref_percentage=model_args.audio_ref_percentage,
                num_codebooks=model_args.num_codebooks,
                audio_encoder_bos_token_id=audio_encoder_bos_token_id,
                audio_encoder_eos_token_id=audio_encoder_eos_token_id,
                use_same_file_ref=data_args.finetune_use_same_file_ref,
                use_precomputed_ref_embed=data_args.finetune_use_precomputed_ref_embed,
                precomputed_ref_embed_path=data_args.finetune_precomputed_ref_embed_path,
            )

            if data_args.max_train_samples is not None:
                indices = random.sample(range(len(train_dataset_local)), data_args.max_train_samples)
                train_dataset_local = Subset(train_dataset_local, indices)
                # train_dataset_local = train_dataset_local.select(range(data_args.max_train_samples))
            sampler = None
            # if training_args.group_by_length:
            #     sampler = LengthGroupedSampler(
            #         training_args.per_device_train_batch_size * accelerator.num_processes,
            #         lengths=train_dataset_local["target_length"],
            #     )
            train_dataloader = DataLoader(
                train_dataset_local,
                collate_fn=data_collator,
                batch_size=training_args.per_device_train_batch_size,
                shuffle=True,
                sampler=sampler,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )

    if training_args.do_eval:
        if data_args.use_mds:
            validation_dataloader = DataLoaderMDS(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                manifest_path=data_args.mds_eval_manifest_path,
                batch_size=training_args.per_device_eval_batch_size,
                prompt_tokenizer=prompt_tokenizer,
                audio_encoder_bos_token_id=audio_encoder_bos_token_id,
                audio_encoder_eos_token_id=audio_encoder_eos_token_id,
                shuffle=False,
                epoch_size=data_args.max_eval_samples,
                collator=data_collator,
                drop_last=True,
            )

        else:
            valid_dataset_local = DatasetLocal(
                root_audio_dir=data_args.finetune_audio_dir,
                root_dac_dir=data_args.finetune_code_dir,
                metadata_path=data_args.finetune_eval_metadata_path,
                prompt_tokenizer=prompt_tokenizer,
                audio_sr=model_args.audio_ref_encoder_sr,
                audio_ref_len=model_args.audio_ref_len,
                audio_ref_percentage=model_args.audio_ref_percentage,
                num_codebooks=model_args.num_codebooks,
                audio_encoder_bos_token_id=audio_encoder_bos_token_id,
                audio_encoder_eos_token_id=audio_encoder_eos_token_id,
                use_same_file_ref=data_args.finetune_use_same_file_ref,
                use_precomputed_ref_embed=data_args.finetune_use_precomputed_ref_embed,
                precomputed_ref_embed_path=data_args.finetune_precomputed_ref_embed_path,
            )

            if data_args.max_eval_samples is not None:
                indices = random.sample(range(len(valid_dataset_local)), data_args.max_eval_samples)
                valid_dataset_local = Subset(valid_dataset_local, indices)

            validation_dataloader = DataLoader(
                valid_dataset_local,
                collate_fn=data_collator,
                batch_size=training_args.per_device_eval_batch_size,
                drop_last=True,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )

    if training_args.predict_with_generate:
        if data_args.use_mds:
            generate_dataloader = DataLoaderMDS(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                manifest_path=data_args.mds_generate_manifest_path,
                batch_size=data_args.per_device_generate_batch_size,
                prompt_tokenizer=prompt_tokenizer,
                audio_encoder_bos_token_id=audio_encoder_bos_token_id,
                audio_encoder_eos_token_id=audio_encoder_eos_token_id,
                shuffle=False,
                epoch_size=data_args.max_generate_samples,
                collator=data_collator,
                drop_last=True,
            )
        else:
            generate_dataset_local = DatasetLocal(
                root_audio_dir=data_args.finetune_audio_dir,
                root_dac_dir=data_args.finetune_code_dir,
                metadata_path=data_args.finetune_generate_metadata_path,
                prompt_tokenizer=prompt_tokenizer,
                audio_sr=model_args.audio_ref_encoder_sr,
                audio_ref_len=model_args.audio_ref_len,
                audio_ref_percentage=model_args.audio_ref_percentage,
                num_codebooks=model_args.num_codebooks,
                audio_encoder_bos_token_id=audio_encoder_bos_token_id,
                audio_encoder_eos_token_id=audio_encoder_eos_token_id,
                use_same_file_ref=data_args.finetune_use_same_file_ref,
                use_precomputed_ref_embed=data_args.finetune_use_precomputed_ref_embed,
                precomputed_ref_embed_path=data_args.finetune_precomputed_ref_embed_path,
            )

            if data_args.max_generate_samples is not None:
                indices = random.sample(range(len(generate_dataset_local)), data_args.max_generate_samples)
                generate_dataset_local = Subset(generate_dataset_local, indices)

            generate_dataloader = DataLoader(
                generate_dataset_local,
                collate_fn=data_collator,
                batch_size=data_args.per_device_generate_batch_size,
                drop_last=True,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )

    # Define Training Schedule
    # Store some constants
    train_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        # steps_per_epoch = len(train_dataset) // (train_batch_size * gradient_accumulation_steps)
        steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info("eval_steps is not set, evaluating at the end of each epoch")
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # T5 doesn't support fp16
    autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))

    # Define optimizer, LR scheduler, collator
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps) * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    # Test the dataloaders
    logger.info("Testing dataloaders")
    if training_args.do_train:
        logger.info(f"Number of training samples: {len(train_dataloader)}")
        for batch in train_dataloader:
            break
        logger.info("Training data example")
        for key, value in batch.items():
            logger.info(f"{key}: {value.shape}")
    if training_args.do_eval:
        logger.info(f"Number of validation samples: {len(validation_dataloader)}")
        for batch in validation_dataloader:
            break
        logger.info("Validation data example")
        for key, value in batch.items():
            logger.info(f"{key}: {value.shape}")
    if training_args.predict_with_generate:
        logger.info(f"Number of generation samples: {len(generate_dataloader)}")
        for batch in generate_dataloader:
            break
        logger.info("Generation data example")
        for key, value in batch.items():
            logger.info(f"{key}: {value.shape}")

    # Prepare everything with accelerate
    logger.info("Preparing model, optimizer and scheduler with accelerate")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if not data_args.use_mds:
        train_dataloader = accelerator.prepare(train_dataloader)
        validation_dataloader = accelerator.prepare(validation_dataloader)
        generate_dataloader = accelerator.prepare(generate_dataloader)

        # Test the dataloaders
        logger.info("AFTER preparing with accelerate")
        logger.info("Testing dataloaders")
        if training_args.do_train:
            logger.info(f"Number of training samples: {len(train_dataloader)}")
            for batch in train_dataloader:
                break
            logger.info("Training data example")
            for key, value in batch.items():
                logger.info(f"{key}: {value.shape}")
        if training_args.do_eval:
            logger.info(f"Number of validation samples: {len(validation_dataloader)}")
            for batch in validation_dataloader:
                break
            logger.info("Validation data example")
            for key, value in batch.items():
                logger.info(f"{key}: {value.shape}")
        if training_args.predict_with_generate:
            logger.info(f"Number of generation samples: {len(generate_dataloader)}")
            for batch in generate_dataloader:
                break
            logger.info("Generation data example")
            for key, value in batch.items():
                logger.info(f"{key}: {value.shape}")

    # Define gradient update step fn

    # def get_ref_embeddings(batch, accelerator):
    #     with accelerator.autocast(autocast_handler=autocast_kwargs):
    #         with torch.no_grad():
    #             encoder_outputs = audio_ref_encoder(batch["audio_ref"], batch["audio_ref_attention_mask"])
    #         if model_args.audio_ref_encoder_hidden_layer is not None:
    #             hidden_layer = model_args.audio_ref_encoder_hidden_layer
    #             encoder_outputs = encoder_outputs.hidden_states[hidden_layer]
    #         else:
    #             encoder_outputs = encoder_outputs.last_hidden_state
    #         if model_args.audio_ref_encoder_mean_pooling:
    #             encoder_outputs = torch.mean(encoder_outputs, dim=1)
    #         encoder_outputs = BaseModelOutput(encoder_outputs)
    #         # Size of encoder_outputs.last_hidden_state is (batch_size, audio_ref length / downsampling, hidden_size)
    #         # Check that batch["attention_mask"] is the size as encoder_outputs and crop/pad as necessary
    #         if "attention_mask" in batch and not model_args.audio_ref_encoder_mean_pooling:
    #             attention_mask = batch["attention_mask"]
    #             encoder_outputs_len = encoder_outputs.last_hidden_state.size(1)
    #             attention_mask_len = attention_mask.size(1)
    #             # attention_mask shape is (batch_size, 1, audio_ref length / downsampling)
    #             # however, this mask isn't always exactly the same length as the encoder_outputs
    #             if encoder_outputs_len < attention_mask_len:
    #                 attention_mask = attention_mask[:, :encoder_outputs_len]
    #             if encoder_outputs_len > attention_mask_len:
    #                 pad_length = encoder_outputs_len - attention_mask_len
    #                 pad = torch.zeros(attention_mask.size(0), pad_length, device=accelerator.device)
    #                 attention_mask = torch.cat([attention_mask, pad], dim=-1)
    #         else:
    #             attention_mask = None
    #     return encoder_outputs, attention_mask

    def get_ref_embeddings(batch, accelerator):
        with accelerator.autocast(autocast_handler=autocast_kwargs):
            with torch.no_grad():
                encoder_outputs = audio_ref_encoder(batch["audio_ref"], batch["audio_ref_attention_mask"])
                # last_hidden_state, extract_features, hidden_states, attentions
            if model_args.audio_ref_encoder_hidden_layer is not None:
                hidden_layer = model_args.audio_ref_encoder_hidden_layer
                if model_args.audio_ref_encoder_mean_pooling:
                    # Updated to use the attentions from the audio ref encoder to create a mask
                    # and ensure that the mean pooling is not including padded positions
                    # This mask is similar to the batch["attention_mask"] but slightly more accurate
                    # Either way, previously we weren't using any attention mask when mean pooling
                    # which was a mistake
                    encoder_output_attentions = encoder_outputs["attentions"][hidden_layer]
                    # batch_size, num_heads, sequence_length, sequence_length
                    encoder_output_attentions = encoder_output_attentions[:, 0, 0, :]
                    # batch size, length
                    encoder_output_attention_mask = torch.where(
                        encoder_output_attentions > 0, torch.tensor(1), torch.tensor(0)
                    )
                    encoder_output_attention_mask = encoder_output_attention_mask.to(accelerator.device)
                    encoder_output_attention_mask = encoder_output_attention_mask.unsqueeze(-1)
                    # batch size, length, 1

                    encoder_outputs = encoder_outputs.hidden_states[hidden_layer]
                    masked_embeddings = encoder_outputs * encoder_output_attention_mask
                    # batch size, length, hidden size

                    # Sum the embeddings along the length dimension
                    sum_embeddings = masked_embeddings.sum(dim=1)  # batch size, embedding_dim

                    # Sum the attention mask along the length dimension to count non-padded positions
                    non_padded_counts = encoder_output_attention_mask.sum(dim=1)  # batch size, 1

                    # Prevent division by zero by setting zero counts to one
                    non_padded_counts = non_padded_counts.masked_fill(non_padded_counts == 0, 1.0)

                    # Divide summed embeddings by the count of non-padded positions to get the mean
                    mean_embeddings = sum_embeddings / non_padded_counts  # batch_size, embedding_dim

                    # Old method below:
                    # encoder_outputs = torch.mean(encoder_outputs, dim=1)
                    encoder_outputs = BaseModelOutput(mean_embeddings)

                else:
                    encoder_outputs = encoder_outputs.hidden_states[hidden_layer]
                    encoder_outputs = BaseModelOutput(encoder_outputs)
            else:
                if model_args.audio_ref_encoder_mean_pooling:
                    raise ValueError("Not currently supported")
                else:
                    encoder_outputs = encoder_outputs.last_hidden_state
                    encoder_outputs = BaseModelOutput(encoder_outputs)

            # Size of encoder_outputs.last_hidden_state is (batch_size, audio_ref length / downsampling, hidden_size)
            # Check that batch["attention_mask"] is the size as encoder_outputs and crop/pad as necessary
            if "attention_mask" in batch and not model_args.audio_ref_encoder_mean_pooling:
                attention_mask = batch["attention_mask"]
                encoder_outputs_len = encoder_outputs.last_hidden_state.size(1)
                attention_mask_len = attention_mask.size(1)
                # attention_mask shape is (batch_size, 1, audio_ref length / downsampling)
                # however, this mask isn't always exactly the same length as the encoder_outputs
                if encoder_outputs_len < attention_mask_len:
                    attention_mask = attention_mask[:, :encoder_outputs_len]
                if encoder_outputs_len > attention_mask_len:
                    pad_length = encoder_outputs_len - attention_mask_len
                    pad = torch.zeros(attention_mask.size(0), pad_length, device=accelerator.device)
                    attention_mask = torch.cat([attention_mask, pad], dim=-1)
            else:
                attention_mask = None
        return encoder_outputs, attention_mask

    def train_step(
        model,
        batch,
        accelerator,
        autocast_kwargs,
    ):
        model.train()

        # TODO - move this "to device" eleswhere
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(accelerator.device)
        # Check if batch["encoder_outputs"] exists in the dictionary, if not, get ref embeddings
        if "encoder_outputs" not in batch:
            encoder_outputs, attention_mask = get_ref_embeddings(batch, accelerator)
            batch["encoder_outputs"] = encoder_outputs
            batch["attention_mask"] = attention_mask
        else:
            batch["encoder_outputs"] = BaseModelOutput(batch["encoder_outputs"])
        outputs = model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss}
        return ce_loss, metrics

    # Define eval fn
    def eval_step(
        model,
        batch,
        accelerator,
        autocast_kwargs,
    ):
        if training_args.torch_compile:
            model = model._orig_mod

        model.eval()
        # TODO - move this "to device" eleswhere
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(accelerator.device)

        if "encoder_outputs" not in batch:
            encoder_outputs, attention_mask = get_ref_embeddings(batch, accelerator)
            batch["encoder_outputs"] = encoder_outputs
            batch["attention_mask"] = attention_mask
        else:
            batch["encoder_outputs"] = BaseModelOutput(batch["encoder_outputs"])
        with torch.no_grad():
            outputs = model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss}

        return metrics

    def generate_step(
        model,
        batch,
        accelerator,
        autocast_kwargs,
    ):
        if training_args.torch_compile:
            model = model._orig_mod

        model = accelerator.unwrap_model(model, keep_fp32_wrapper=mixed_precision != "fp16")
        model.eval()
        # TODO - move this "to device" eleswhere
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(accelerator.device)
        if "encoder_outputs" not in batch:
            encoder_outputs, attention_mask = get_ref_embeddings(batch, accelerator)
            batch["encoder_outputs"] = encoder_outputs
            batch["attention_mask"] = attention_mask
        else:
            batch["encoder_outputs"] = BaseModelOutput(batch["encoder_outputs"])

        audio_refs = batch.pop("audio_ref", None)
        # audio_refs are padded with -10000000 (to ensure safe attention mask creation), replace with 0s
        audio_refs = torch.where(audio_refs == -10000000, torch.tensor(0, device=accelerator.device), audio_refs)
        batch.pop("audio_ref_attention_mask", None)
        batch.pop("decoder_attention_mask", None)

        output_audios = model.generate(**batch, **gen_kwargs)
        output_audios = accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios, audio_refs

    logger.info("***** Running training *****")
    logger.info(f"  Num samples in training dataset= {(len(train_dataloader) * train_batch_size):,}")
    logger.info(f"  Num examples = {(total_train_steps * train_batch_size * gradient_accumulation_steps):,}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps:,}")
    if data_args.finetune_use_precomputed_ref_embed:
        logger.info("  Using precomputed reference embeddings")
    if data_args.finetune_precomputed_ref_embed_path is not None:
        logger.info(f"  Loading precomputed reference embeddings from {data_args.finetune_precomputed_ref_embed_path}")
    if "attention_mask" in batch and not model_args.audio_ref_encoder_mean_pooling:
        logger.info("  Using attention mask for audio ref encoder")
    else:
        logger.info("  Not using attention mask for audio ref encoder")
    if model_args.audio_ref_encoder_hidden_layer is not None:
        logger.info(f"  Using hidden layer {model_args.audio_ref_encoder_hidden_layer} for audio ref encoder")
    if model_args.audio_ref_encoder_mean_pooling:
        logger.info("  Using mean pooling for audio ref encoder")
    if data_args.finetune_use_precomputed_ref_embed:
        logger.info("  Using precomputed reference embeddings")
    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        if training_args.push_to_hub:
            logger.info("Pushing to the hub")
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            logger.info(f"Creating model checkpoint output directory: {training_args.output_dir}")
            os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info("Accelerator waiting for everyone")
    accelerator.wait_for_everyone()

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with accelerator.main_process_first():
        logger.info("Saving model checkpoint and configuration to output directory...")
        # only the main process saves them
        if accelerator.is_main_process:
            # save tokenizer and config
            if (
                model_args.prompt_tokenizer_name is None
                and model_args.description_tokenizer_name
                or (
                    model_args.prompt_tokenizer_name == model_args.description_tokenizer_name
                )  # NOTE not actually using description tokenizer
            ):
                prompt_tokenizer.save_pretrained(training_args.output_dir)
            else:
                logger.warning(
                    f"Prompt tokenizer ('{model_args.prompt_tokenizer_name}') and description tokenizer ('{model_args.description_tokenizer_name}') are not the same. Saving only the prompt tokenizer."
                )
                prompt_tokenizer.save_pretrained(training_args.output_dir)

            config.save_pretrained(training_args.output_dir)

    if checkpoint is not None:
        logger.info(f"Checkpoint found, loading: {checkpoint}")
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
    else:
        resume_step = None

    gen_kwargs = {
        "do_sample": model_args.do_sample,
        "temperature": model_args.temperature,
        "max_length": model_args.max_length,
        # Because of the delayed pattern mask, generation might stop earlier because of unexpected behaviour
        # on the first tokens of the codebooks that are delayed.
        # This fix the issue.
        "min_new_tokens": model_args.num_codebooks + 1,
    }

    logger.info(f"Updated gen_kwargs: {gen_kwargs}")

    for epoch in range(epochs_trained, num_epochs):
        # TODO Check the below
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            logger.info(f"NOT skipping the first {resume_step} batches in the dataloader (very slow, revisit this)")
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            # train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        logger.info(f"Running training epoch {epoch}")
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, train_metric = train_step(model, batch, accelerator, autocast_kwargs)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    logger.info(f"Saving checkpoint for step {cur_step} at epoch {epoch}")
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    # safe_serialization=False to avoid shared tensors saving issue (TODO(YL): it's a temporary fix)
                    # https://github.com/huggingface/transformers/issues/27293#issuecomment-1872560074
                    accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(
                            training_args.save_total_limit, output_dir=training_args.output_dir, logger=logger
                        )

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(training_args.output_dir)

                        if training_args.push_to_hub:
                            logger.info("Pushing to the hub...")
                            repo.push_to_hub(
                                commit_message=f"Saving train state of step {cur_step}",
                                blocking=False,
                            )

                # if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                if training_args.do_eval and (
                    cur_step % eval_steps == 0 or cur_step == total_train_steps or cur_step == 1
                ):
                    # ======================== Evaluating ==============================
                    logger.info("***** Running evaluation *****")
                    train_time += time.time() - train_start
                    eval_metrics = []
                    eval_preds = []
                    eval_refs = []
                    eval_prompts = []
                    eval_start = time.time()
                    # release training input batch
                    batch = release_memory(batch)

                    for batch in tqdm(
                        validation_dataloader,
                        desc="Evaluating - Inference ...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Model forward
                        eval_metric = eval_step(model, batch, accelerator, autocast_kwargs)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)

                    if training_args.predict_with_generate:
                        generation_count = 0
                        logger.info("***** Running generation *****")
                        # release eval input batch (in favour of generate)
                        batch = release_memory(batch)
                        # generation
                        for batch in tqdm(
                            generate_dataloader,
                            desc="Evaluating - Generation ...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                        ):
                            generated_audios, audio_refs = generate_step(model, batch, accelerator, autocast_kwargs)
                            # Gather all predictions and targets
                            generated_audios, audio_refs, prompts = accelerator.pad_across_processes(
                                (generated_audios, audio_refs, batch["prompt_input_ids"]), dim=1, pad_index=0
                            )
                            generated_audios, audio_refs, prompts = accelerator.gather_for_metrics(
                                (generated_audios, audio_refs, prompts)
                            )
                            eval_preds.extend(generated_audios)
                            eval_prompts.extend(prompts.to("cpu"))
                            eval_refs.extend(audio_refs)
                            generation_count += len(generated_audios)
                            if generation_count >= 100:  # TODO remove hard-coded value (wandb can only do 100)
                                break

                    eval_time = time.time() - eval_start
                    # normalize eval metrics
                    eval_metrics = {
                        key: torch.mean(torch.cat([d[key].unsqueeze(0) for d in eval_metrics]))
                        for key in eval_metrics[0]
                    }

                    # compute metrics
                    metrics_desc = ""
                    if training_args.predict_with_generate:
                        # Just use the main process to compute the metrics
                        metric_values, pred_prompts, transcriptions = compute_metrics(
                            eval_preds,
                            eval_refs,
                            eval_prompts,
                            model_args.asr_model_name_or_path,
                            data_args.per_device_generate_batch_size,
                            prompt_tokenizer,
                            sample_rate,
                            model_args.audio_ref_encoder_sr,
                            accelerator.device,
                        )

                        eval_metrics.update(metric_values)
                        metrics_desc = " ".join([f"Eval {key}: {value} |" for key, value in metric_values.items()])
                        if "wandb" in training_args.report_to:
                            log_pred(
                                accelerator,
                                pred_prompts,
                                transcriptions,
                                eval_preds,
                                eval_refs,
                                sampling_rate=sample_rate,
                                step=cur_step,
                                prefix="eval",
                            )

                    # Print metrics and update progress bar
                    steps_trained_progress_bar.write(
                        f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                        f" {metrics_desc})"
                    )

                    log_metric(
                        accelerator,
                        metrics=eval_metrics,
                        train_time=eval_time,
                        step=cur_step,
                        epoch=epoch,
                        prefix="eval",
                    )

                    # release eval batch and relax metrics
                    eval_metrics = []
                    eval_preds = []
                    eval_prompts = []
                    batch = release_memory(batch)

                    # flush the train metrics
                    train_start = time.time()

                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    set_start_method("spawn")
    main()
