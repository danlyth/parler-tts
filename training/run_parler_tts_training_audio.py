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

""" Train Parler-TTS using 🤗 Accelerate"""

import logging
import os
import re
import sys
import time
from multiprocess import set_start_method
from datetime import timedelta
import random

from tqdm import tqdm
from pathlib import Path

import torch    
from torch.utils.data import DataLoader, Subset

import datasets
from datasets import DatasetDict, IterableDataset

from huggingface_hub import Repository, create_repo
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    AutoModel,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry

from accelerate import Accelerator
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin
from accelerate.utils.memory import release_memory

from parler_tts import (
    ParlerTTSForConditionalGeneration,
    ParlerTTSConfig,
)

from training.utils import get_last_checkpoint, rotate_checkpoints, log_pred, log_metric
from training.arguments import ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments
from training.data_local import DatasetLocal, DataCollator
from training.data_mds import DatasetMDS, gather_streams, configure_aws_creds
from training.eval import clap_similarity, wer


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

    if data_args.use_mds:
        configure_aws_creds()


    if training_args.dtype == "float16":
        mixed_precision = "fp16"
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
    else:
        mixed_precision = "no"

    if data_args.pad_to_max_length and (
        data_args.max_duration_in_seconds is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "`pad_to_max_length` is `True` but one of the following parameters has not been set: `max_duration_in_seconds`, `max_prompt_token_length`, `max_description_token_length`"
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
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "freeze_text_encoder": model_args.freeze_text_encoder,
            "max_duration_in_seconds": data_args.max_duration_in_seconds,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": model_args.temperature,
        },
    )

    

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

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
    num_workers = data_args.preprocessing_num_workers

    # 1. First, let's instantiate the feature extractor (DAC), tokenizers and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    sample_rate = model_args.discrete_audio_feature_sample_rate # TODO (Dan) need to get this from somewhere else as I won't be using the feature extractor

    # load prompt tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",  # prompt has to be padded on the left bc it's preprend to codebooks hidden states
    )

    # load description tokenizer # TODO (Dan) remove this if I decide not to use it at all
    description_tokenizer = AutoTokenizer.from_pretrained(
        model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )

    if model_args.use_fast_tokenizer:
        logger.warning(
            "Disabling fast tokenizer warning: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3231-L3235"
        )
        prompt_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        description_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # load audio reference encoder
    audio_ref_encoder = AutoModel.from_pretrained(model_args.audio_ref_encoder_name)
    audio_ref_encoder.to(training_args.device)


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
            "pad_token_id": model_args.pad_token_id
            if model_args.pad_token_id is not None
            else config.pad_token_id,
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

    audio_max_length = None
    if training_args.torch_compile: # TODO (Dan) check this works
        audio_max_length = max(vectorized_datasets["train"]["target_length"])
        with accelerator.main_process_first():
            max_sample = vectorized_datasets["train"].filter(
                lambda x: x == audio_max_length,
                num_proc=num_workers,
                input_columns=["target_length"],
            )
        audio_max_length = torch.tensor(max_sample[0]["labels"]).shape[1]

    audio_encoder_eos_token_id = config.decoder.eos_token_id
    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id

    # 2. Now, let's load the dataset # TODO (Dan) change numbering
    vectorized_datasets = DatasetDict()

    # TODO (Dan) remove all this hard-coding
    if training_args.do_train:

        if data_args.use_mds:
            streams = gather_streams(data_args.mds_train_manifest_path,
                                     data_args.mds_s3_bucket_root,
                                     data_args.mds_cache_dir)
            vectorized_datasets["train"] = DatasetMDS(streams=streams,
                                                      batch_size=training_args.per_device_train_batch_size,
                                                      prompt_tokenizer=prompt_tokenizer,
                                                      audio_sample_rate=model_args.audio_ref_encoder_sr,
                                                      audio_ref_len=model_args.audio_ref_len,
                                                      shuffle=True,
                                                      cache_limit=data_args.mds_cache_limit,
                                                      )

        else:
            vectorized_datasets["train"] = DatasetLocal(
            # root_audio_dir=data_args.root_audio_dir,
            root_audio_dir="/data/expresso/audio_48khz_short_chunks_ex02_processed",
            # root_dac_dir=data_args.root_dac_dir,
            root_dac_dir="/data/expresso/audio_48khz_short_chunks_ex02_processed/dac_codes",
            # metadata_path=data_args.train_metadata_path,
            metadata_path="/data/expresso/audio_48khz_short_chunks_ex02_processed/train_local.tsv",
            prompt_tokenizer=prompt_tokenizer,
            audio_sr=16000, # TODO (Dan) remove these three lines of hard-coding
            audio_ref_len=2,
            num_codebooks=9,
            audio_encoder_bos_token_id=audio_encoder_bos_token_id,
            audio_encoder_eos_token_id=audio_encoder_eos_token_id,
        )

            # TODO (Dan) check this works
            if data_args.max_train_samples is not None:
                indices = random.sample(range(len(vectorized_datasets["train"])), data_args.max_train_samples)
                vectorized_datasets["train"] = Subset(vectorized_datasets["train"], indices)
                # vectorized_datasets["train"] = vectorized_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:

        if data_args.use_mds:
            streams = gather_streams(data_args.mds_eval_manifest_path,
                                     data_args.mds_s3_bucket_root,
                                     data_args.mds_cache_dir)
            vectorized_datasets["eval"] = DatasetMDS(streams=streams,
                                                     batch_size=training_args.per_device_eval_batch_size,
                                                     prompt_tokenizer=prompt_tokenizer,
                                                     audio_sample_rate=model_args.audio_ref_encoder_sr,
                                                     audio_ref_len=model_args.audio_ref_len,
                                                     shuffle=False,
                                                     cache_limit=data_args.mds_cache_limit,
                                                     )

            # TODO (Dan) figure out how select particular number of eval samples with MDS

        else:
            vectorized_datasets["eval"] = DatasetLocal(
            # root_audio_dir=data_args.root_audio_dir,
            root_audio_dir="/data/expresso/audio_48khz_short_chunks_ex02_processed",
            # root_dac_dir=data_args.root_dac_dir,
            root_dac_dir="/data/expresso/audio_48khz_short_chunks_ex02_processed/dac_codes",
            # metadata_path=data_args.eval_metadata_path,
            metadata_path="/data/expresso/audio_48khz_short_chunks_ex02_processed/dev_local.tsv",
            prompt_tokenizer=prompt_tokenizer,
            audio_sr=16000, # TODO (Dan) remove these three lines of hard-coding
            audio_ref_len=2,
            num_codebooks=9,
            audio_encoder_bos_token_id=audio_encoder_bos_token_id,
            audio_encoder_eos_token_id=audio_encoder_eos_token_id,
    )

            # TODO (Dan) check this works
            if data_args.max_eval_samples is not None:
                indices = random.sample(range(len(vectorized_datasets["eval"])), data_args.max_eval_samples)
                vectorized_datasets["eval"] = Subset(vectorized_datasets["eval"], indices)

    if training_args.predict_with_generate:

        if data_args.use_mds:
            streams = gather_streams(data_args.mds_generate_manifest_path,
                                     data_args.mds_s3_bucket_root,
                                     data_args.mds_cache_dir)
            vectorized_datasets["generate"] = DatasetMDS(streams=streams,
                                                         batch_size=training_args.per_device_predict_batch_size,
                                                         prompt_tokenizer=prompt_tokenizer,
                                                         audio_sample_rate=model_args.audio_ref_encoder_sr,
                                                         audio_ref_len=model_args.audio_ref_len,
                                                         shuffle=False,
                                                         cache_limit=data_args.mds_cache_limit,
                                                         epoch=data_args.max_generate_samples
                                                         )

            # TODO (Dan) figure out how select particular number of predict samples with MDS


    # 6. Next, we can prepare the training.

    # Let's use word CLAP similary and WER metrics as our evaluation metrics
    # TODO (Dan) Move this to eval
    def compute_metrics(audios,
                        descriptions,
                        prompts,
                        asr_model_name_or_path,
                        clap_model_name_or_path,
                        batch_size,
                        description_tokenizer,
                        prompt_tokenizer,
                        sample_rate,
                        device="cpu"
                        ):
        results = {}
        input_ids = descriptions
        texts = description_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompts = prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)
        audios = [a.cpu().numpy() for a in audios]
        
        clap_score = clap_similarity(clap_model_name_or_path, texts, audios, device)
        results["clap"] = clap_score

        word_error, transcriptions = wer(asr_model_name_or_path,
                                         prompts,
                                         audios,
                                         device,
                                         batch_size,
                                         sample_rate)
        results["wer"] = word_error

        return results, texts, prompts, audios, transcriptions

    # Define Training Schedule
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info(f"eval_steps is not set, evaluating at the end of each epoch")
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

    # Instantiate custom data collator
    # TODO (Dan) decide on whether we want description conditioning
    data_collator = DataCollator(
        prompt_tokenizer=prompt_tokenizer,
        # description_tokenizer=description_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        # description_max_length=data_args.max_description_token_length,
        audio_max_length=audio_max_length,
    )

    # Prepare everything with accelerate
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device =" f" {per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

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
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with accelerator.main_process_first():
        # only the main process saves them
        if accelerator.is_main_process:
            # save feature extractor, tokenizer and config
            if (
                model_args.prompt_tokenizer_name is None
                and model_args.description_tokenizer_name
                or (model_args.prompt_tokenizer_name == model_args.description_tokenizer_name)
            ):
                prompt_tokenizer.save_pretrained(training_args.output_dir)
            else:
                logger.warning(
                    f"Prompt tokenizer ('{model_args.prompt_tokenizer_name}') and description tokenizer ('{model_args.description_tokenizer_name}') are not the same. Saving only the prompt tokenizer."
                )
                prompt_tokenizer.save_pretrained(training_args.output_dir)

            # feature_extractor.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    if checkpoint is not None:
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

        # TODO (Dan) fix shuffling so that you can continue from a checkpoint properly
        # for epoch in range(0, epochs_trained):
            # vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            # TODO (Dan) fix shuffling
            # vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    gen_kwargs = {
        "do_sample": model_args.do_sample,
        "temperature": model_args.temperature,
        "max_length": model_args.max_length,
    }

    # Define gradient update step fn
    def train_step(
        batch,
        accelerator,
        autocast_kwargs,
    ):
        model.train()

        with accelerator.autocast(autocast_handler=autocast_kwargs):
            if training_args.parallel_mode.value != "distributed": # TODO (Dan) check we're supporting this properly
                with torch.no_grad(): # Does autocast take care of this?
                    encoder_outputs = audio_ref_encoder(batch["audio_ref"]) # TODO move this to dataset?
            else:
                with torch.no_grad(): # Does autocast take care of this?
                    encoder_outputs = audio_ref_encoder(batch["audio_ref"])
            batch["encoder_outputs"] = encoder_outputs # Size (batch_size, audio_ref length / downsampling, hidden_size)
            # batch["attention_mask"] = torch.ones_like(batch["encoder_outputs"]) # Can do this because we're not using padding

        outputs = model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss

        metrics = {"loss": ce_loss}
        return ce_loss, metrics

    # Define eval fn
    def eval_step(
        batch,
        accelerator,
        autocast_kwargs,
    ):
        eval_model = model if not training_args.torch_compile else model._orig_mod
        eval_model.eval()

        with accelerator.autocast(autocast_handler=autocast_kwargs):
            if training_args.parallel_mode.value != "distributed": # TODO check we're supporting this properly
                encoder_outputs = audio_ref_encoder(batch["audio_ref"])
            else:
                encoder_outputs = audio_ref_encoder(batch["audio_ref"])
            batch["encoder_outputs"] = encoder_outputs
            # batch["attention_mask"] = torch.ones_like(batch["encoder_outputs"])

        with torch.no_grad():
            outputs = eval_model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss}
        return metrics

    def generate_step(batch):
        with accelerator.autocast(autocast_handler=autocast_kwargs):
            if training_args.parallel_mode.value != "distributed": # TODO check we're supporting this properly
                encoder_outputs = audio_ref_encoder(batch["audio_ref"])
            else:
                encoder_outputs = audio_ref_encoder(batch["audio_ref"])
            batch["encoder_outputs"] = encoder_outputs
            # batch["attention_mask"] = torch.ones_like(batch["encoder_outputs"])

        batch.pop("audio_ref", None)
        batch.pop("audio_ref_attention_mask", None)


        batch.pop("decoder_attention_mask", None)
        eval_model = accelerator.unwrap_model(model, keep_fp32_wrapper=mixed_precision != "fp16").eval()
        if training_args.torch_compile:
            eval_model = model._orig_mod

        output_audios = eval_model.generate(**batch, **gen_kwargs)
        output_audios = accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios

    for epoch in range(epochs_trained, num_epochs):
        sampler = None
        # TODO (Dan) fix sampler if needed
        # if training_args.group_by_length:
        #     sampler = LengthGroupedSampler(train_batch_size, lengths=vectorized_datasets["train"]["target_length"])
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            shuffle=True, # TODO (Dan) use seed?
            sampler=sampler,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, train_metric = train_step(batch, accelerator, autocast_kwargs)
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
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    # safe_serialization=False to avoid shared tensors saving issue (TODO(YL): it's a temporary fix)
                    # https://github.com/huggingface/transformers/issues/27293#issuecomment-1872560074
                    accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir, logger=logger)

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(training_args.output_dir)

                        if training_args.push_to_hub:
                            repo.push_to_hub(
                                commit_message=f"Saving train state of step {cur_step}",
                                blocking=False,
                            )

                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps or cur_step == 1):
                    # TODO (Dan) I added this condition to evaluate at the first step, might not want it after debugging
                    train_time += time.time() - train_start
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_preds = []
                    eval_descriptions = []
                    eval_prompts = []
                    eval_start = time.time()

                    # release training input batch
                    batch = release_memory(batch)

                    validation_dataloader = DataLoader(
                        vectorized_datasets["eval"],
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=training_args.dataloader_pin_memory,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating - Inference ...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Model forward
                        eval_metric = eval_step(batch, accelerator, autocast_kwargs)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)

                    if training_args.predict_with_generate:

                        # generation
                        for batch in tqdm(
                            validation_dataloader,
                            desc=f"Evaluating - Generation ...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                        ):
                            generated_audios = generate_step(batch)
                            # Gather all predictions and targets
                            generated_audios, prompts = accelerator.pad_across_processes(
                                (generated_audios, batch["prompt_input_ids"]), dim=1, pad_index=0
                            )
                            generated_audios, prompts = accelerator.gather_for_metrics(
                                (generated_audios, prompts)
                            )
                            eval_preds.extend(generated_audios.to("cpu"))
                            # eval_descriptions.extend(input_ids.to("cpu")) 
                            eval_descriptions.extend("No description") # TODO (Dan) fix this hard-coding
                            eval_prompts.extend(prompts.to("cpu"))

                    eval_time = time.time() - eval_start
                    # normalize eval metrics
                    eval_metrics = {
                        key: torch.mean(torch.cat([d[key].unsqueeze(0) for d in eval_metrics]))
                        for key in eval_metrics[0]
                    }

                    # compute metrics
                    metrics_desc = ""
                    if training_args.predict_with_generate:
                        # metric_values, pred_descriptions, pred_prompts, audios, transcriptions = compute_metrics(
                        #     eval_preds, eval_descriptions, eval_prompts, accelerator.device
                        # ) # TODO (Dan), tidy this up
                        metric_values, pred_descriptions, pred_prompts, audios, transcriptions = compute_metrics(
                            eval_preds,
                            eval_descriptions,
                            eval_prompts,
                            model_args.asr_model_name_or_path,
                            model_args.clap_model_name_or_path,
                            model_args.per_device_eval_batch_size,
                            description_tokenizer,
                            prompt_tokenizer,
                            sample_rate,
                            accelerator.device
                        )
                        pred_descriptions = ["No description" for _ in range(len(pred_prompts))] # TODO (Dan) remove this at some point
                        eval_metrics.update(metric_values)
                        metrics_desc = " ".join([f"Eval {key}: {value} |" for key, value in metric_values.items()])
                        if "wandb" in training_args.report_to:
                            log_pred(
                                accelerator,
                                pred_descriptions,
                                pred_prompts,
                                transcriptions,
                                audios,
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
                    eval_descriptions = []
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