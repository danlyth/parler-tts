import logging
from pathlib import Path
import random
import sys
import time

import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.modeling_outputs import BaseModelOutput

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
)
from training.arguments import DataTrainingArguments, ModelArguments, ParlerTTSTrainingArguments
from training.data_local import DataCollator, DatasetLocal


def get_ref_embeddings(batch, audio_ref_encoder, model_args, device):
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
            pad = torch.zeros(attention_mask.size(0), pad_length, device=device)
            attention_mask = torch.cat([attention_mask, pad], dim=-1)
    else:
        attention_mask = None
    return encoder_outputs, attention_mask


def main(json_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=Path(json_path).resolve())

    sample_rate = model_args.discrete_audio_feature_sample_rate

    ############################################
    #       Load models and tokenizers         #
    ############################################

    # Load tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",  # prompt has to be padded on the left bc it's preprend to codebooks hidden states
    )

    # Load audio reference encoder
    audio_ref_encoder = AutoModel.from_pretrained(model_args.audio_ref_encoder_name, output_hidden_states=True)
    audio_ref_encoder.to(training_args.device)
    audio_ref_encoder.eval()

    # Load the config
    config = ParlerTTSConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # Update pad token id and decoder_start_token_id
    config.update(
        {
            "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else config.decoder_start_token_id,
        }
    )

    # Load main model
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    gen_kwargs = {
        "do_sample": model_args.do_sample,
        "temperature": model_args.temperature,
        "max_length": model_args.max_length,
        # Because of the delayed pattern mask, generation might stop earlier because of unexpected behaviour
        # on the first tokens of the codebooks that are delayed.
        # This fix the issue.
        "min_new_tokens": model_args.num_codebooks + 1,
    }

    model.to(device)
    model.eval()

    ############################################
    #               Load data                  #
    ############################################

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
    audio_encoder_eos_token_id = config.decoder.eos_token_id

    data_collator = DataCollator(
        prompt_tokenizer=prompt_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        audio_max_length=data_args.max_audio_token_length,
        audio_ref_max_length=data_args.max_audio_ref_length,
    )

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

    ############################################
    #               Generate                   #
    ############################################

    start_time = time.strftime("%Y%m%d-%H%M%S")
    audio_output_dir = Path(model_args.model_name_or_path) / f"{start_time}_generations"
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    for index, batch in tqdm(enumerate(generate_dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        if "encoder_outputs" not in batch:
            encoder_outputs, attention_mask = get_ref_embeddings(batch, audio_ref_encoder, model_args, device)
            batch["encoder_outputs"] = encoder_outputs
            batch["attention_mask"] = attention_mask
        else:
            batch["encoder_outputs"] = BaseModelOutput(batch["encoder_outputs"])

        batch.pop("audio_ref", None)
        batch.pop("audio_ref_attention_mask", None)
        batch.pop("decoder_attention_mask", None)

        output_audios = model.generate(**batch, **gen_kwargs)

        for i, audio in enumerate(output_audios):
            audio_path = audio_output_dir / f"test_output_{index}_{i}.wav"
            torchaudio.save(str(audio_path), audio.cpu().unsqueeze(0), sample_rate=sample_rate)


if __name__ == "__main__":
    main("/home/ankit/code/parler-tts/helpers/training_configs/finetune_maya/maya_inference.json")
