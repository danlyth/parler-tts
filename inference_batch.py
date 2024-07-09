import logging
import os
import random
import re
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, set_seed
from accelerate.utils.memory import release_memory
from multiprocess import set_start_method
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import send_example_telemetry

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
)
from training.arguments import DataTrainingArguments, ModelArguments, ParlerTTSTrainingArguments
from training.data_local import DataCollator, DatasetLocal
from training.eval import compute_metrics
from training.utils import get_last_checkpoint, log_metric, log_pred, rotate_checkpoints


logger = logging.getLogger(__name__)


def main(json_path, temperature_override):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    # NOTE change here to pass json path directly rather than use sys.argv
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(json_path))

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
    # NOTE removed this
    # set_seed(training_args.seed)

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
    audio_ref_encoder = AutoModel.from_pretrained(model_args.audio_ref_encoder_name, output_hidden_states=True)
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

    # NOTE change from here
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
        return_full_ref_audio=model_args.return_full_ref_audio,
    )

    if data_args.max_generate_samples is not None:
        indices = random.sample(range(len(generate_dataset_local)), data_args.max_generate_samples)
        generate_dataset_local = Subset(generate_dataset_local, indices)

    generate_dataloader = DataLoader(
        generate_dataset_local,
        collate_fn=data_collator,
        batch_size=data_args.per_device_generate_batch_size,
        drop_last=False,  # NOTE was True
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    # NOTE change from here

    # T5 doesn't support fp16
    autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))

    logger.info("Testing dataloaders")
    logger.info(f"Number of generation samples: {len(generate_dataloader)}")
    for batch in generate_dataloader:
        break
    logger.info("Generation data example")
    for key, value in batch.items():
        logger.info(f"{key}: {value.shape}")

    # Prepare everything with accelerate
    model = accelerator.prepare(model)
    generate_dataloader = accelerator.prepare(generate_dataloader)

    logger.info("AFTER preparing with accelerate")
    logger.info("Testing dataloaders")
    logger.info(f"Number of generation samples: {len(generate_dataloader)}")
    for batch in generate_dataloader:
        break
    logger.info("Generation data example")
    for key, value in batch.items():
        logger.info(f"{key}: {value.shape}")

    # NOTE moved this up from further below

    # NOTE added this override
    if temperature_override is None:
        temperature = model_args.temperature
    else:
        temperature = temperature_override
    gen_kwargs = {
        "do_sample": model_args.do_sample,
        "temperature": temperature,
        "max_length": model_args.max_length,
        # Because of the delayed pattern mask, generation might stop earlier because of unexpected behaviour
        # on the first tokens of the codebooks that are delayed.
        # This fix the issue.
        "min_new_tokens": model_args.num_codebooks + 1,
    }

    logger.info(f"Updated gen_kwargs: {gen_kwargs}")

    def get_ref_embeddings(batch, accelerator):
        with accelerator.autocast(autocast_handler=autocast_kwargs):
            with torch.no_grad():
                encoder_outputs = audio_ref_encoder(batch["audio_ref"], batch["audio_ref_attention_mask"])
            if model_args.audio_ref_encoder_hidden_layer is not None:
                hidden_layer = model_args.audio_ref_encoder_hidden_layer
                encoder_outputs = encoder_outputs.hidden_states[hidden_layer]
            else:
                encoder_outputs = encoder_outputs.last_hidden_state
            if model_args.audio_ref_encoder_mean_pooling:
                encoder_outputs = torch.mean(encoder_outputs, dim=1)
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
        batch.pop("audio_ref_attention_mask", None)
        batch.pop("decoder_attention_mask", None)
        if "audio_ref_full" in batch:
            batch.pop("audio_ref_full", None)

        output_audios = model.generate(**batch, **gen_kwargs)
        output_audios = accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios, audio_refs

    eval_metrics = []
    eval_preds = []
    eval_refs = []
    eval_prompts = []
    eval_start = time.time()

    # generation_count = 0
    logger.info("***** Running generation *****")
    # only run on main process
    # with accelerator.main_process_first():
    for batch in tqdm(
        generate_dataloader,
        desc="Evaluating - Generation ...",
        position=2,
        disable=not accelerator.is_local_main_process,
    ):
        if "audio_ref_full" in batch:
            audio_refs_full = batch["audio_ref_full"]
        else:
            audio_refs_full = None
        generated_audios, audio_refs = generate_step(model, batch, accelerator, autocast_kwargs)
        # Gather all predictions and targets
        generated_audios, prompts = accelerator.pad_across_processes(
            (generated_audios, batch["prompt_input_ids"]), dim=1, pad_index=0
        )
        generated_audios, prompts = accelerator.gather_for_metrics((generated_audios, prompts))
        eval_preds.extend(generated_audios)
        eval_prompts.extend(prompts.to("cpu"))
        if audio_refs_full is not None:
            eval_refs.extend(audio_refs_full)
        else:
            eval_refs.extend(audio_refs)
        # generation_count += len(generated_audios)
        # if generation_count >= 100:  # TODO remove hard-coded value (wandb can only do 100)
        # break

    eval_time = time.time() - eval_start

    # compute metrics
    if training_args.predict_with_generate:
        # Just use the main process to compute the metrics
        metric_values, pred_prompts, audios, transcriptions = compute_metrics(
            eval_preds,
            eval_refs,
            eval_prompts,
            model_args.asr_model_name_or_path,
            data_args.per_device_generate_batch_size,
            prompt_tokenizer,
            sample_rate,
            accelerator.device,
        )

        eval_metrics = metric_values
        if "wandb" in training_args.report_to:
            log_pred(
                accelerator,
                pred_prompts,
                transcriptions,
                audios,
                eval_refs,
                sampling_rate=sample_rate,
                step=1,
                prefix="eval",
            )

    log_metric(
        accelerator,
        metrics=eval_metrics,
        train_time=eval_time,
        step=1,
        epoch=1,
        prefix="eval",
    )

    # release eval batch and relax metrics
    eval_metrics = []
    eval_preds = []
    eval_prompts = []
    batch = release_memory(batch)

    accelerator.end_training()


if __name__ == "__main__":
    json_path = "/home/ankit/code/parler-tts/helpers/training_configs/finetune_maya/maya_inference.json"
    temperatures = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for temperature in temperatures:
        main(json_path, temperature_override=temperature)
