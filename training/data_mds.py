import io
import os
from pathlib import Path
from typing import Optional
import json

import numpy as np
import streaming
import torch

import torchaudio
from streaming import Stream, StreamingDataset, StreamingDataLoader

from transformers import AutoTokenizer

from parler_tts.modeling_parler_tts import build_delay_pattern_mask
from training.arguments import DataTrainingArguments, ModelArguments, ParlerTTSTrainingArguments


def configure_aws_creds():
    os.environ["S3_ENDPOINT_URL"] = f"https://{os.environ['CF_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CF_AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CF_AWS_SECRET_ACCESS_KEY"]


def gather_streams(manifest_path: str, s3_bucket_root: str, mds_cache_dir: str):
    """Instantiate SpeechDataset from a path to a directory containing a manifest as a txt file.

    Args:
        manifest_path (str or Path): Path to manifest text file containing list of dataset names.
        s3_bucket_root (str): Path to s3 bucket containing the above datasets.
        mds_cache_dir (str or Path): Path to local directory to store the downloaded datasets.
    """

    streaming.base.util.clean_stale_shared_memory()

    mds_cache_dir = Path(mds_cache_dir)

    with open(manifest_path, "r") as f:  # type: ignore
        dataset_names = f.read().splitlines()
    dataset_names = [dataset_name.strip() for dataset_name in dataset_names]
    cache_dirs = [mds_cache_dir / Path(*dataset_name.split("/")) for dataset_name in dataset_names]

    # s3_bucket_root is, for example, s3://my-data-bucket/
    buckets = [str(s3_bucket_root + dataset_name) for dataset_name in dataset_names]
    assert len(buckets) > 0, "No datasets found in manifest."

    # Should now have e.g. ["s3://my-data-bucket/dataset_name_1", "s3://my-data-bucket/dataset_name_2", etc.]
    streams = [Stream(remote=bucket, local=cache_dir) for bucket, cache_dir in zip(buckets, cache_dirs)]

    return streams


class DatasetMDS(StreamingDataset):
    def __init__(
        self,
        streams: list,
        batch_size: int,
        prompt_tokenizer: AutoTokenizer,
        num_codebooks: int,  # number of codebooks in the DAC features
        audio_ref_sample_rate: int,  # audio sample rate for reference encoder, typically 16kHz
        audio_ref_len: Optional[int] = None,  # audio reference length (in seconds) for reference encoder
        audio_ref_percentage: Optional[float] = None,  # percentage of audio to use for reference encoder
        audio_encoder_bos_token_id: int = None,  # BOS token id for audio encoder
        audio_encoder_eos_token_id: int = None,  # EOS token id for audio encoder
        **kwargs,
    ):
        super().__init__(
            streams=streams,  # list of Stream objects
            batch_size=batch_size,  # necessary for deterministic resumption and optimal performance
            #  download_retry=kwargs["retry"], # Default is 2
            #  download_timeout=kwargs["timeout"], # Default is 60
            shuffle=kwargs["shuffle"],  # Default is False
            cache_limit=kwargs["cache_limit"],  # NOTE this isn't working for some reason...
            epoch_size=kwargs[
                "epoch_size"
            ],  # Number of samples to draw per epoch balanced across all streams. Useful for limiting dataset size for debugging or generation
            # See https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html
            #  for full list of available arguments
        )

        self.audio_sr = audio_ref_sample_rate
        self.audio_ref_len = audio_ref_len
        self.audio_ref_percentage = audio_ref_percentage
        self.prompt_tokenizer = prompt_tokenizer
        self.num_codebooks = num_codebooks
        if audio_encoder_bos_token_id is None:
            raise ValueError("audio_encoder_bos_token_id must be provided")
        if audio_encoder_eos_token_id is None:
            raise ValueError("audio_encoder_eos_token_id must be provided")
        self.audio_encoder_bos_token_id = audio_encoder_bos_token_id
        self.audio_encoder_eos_token_id = audio_encoder_eos_token_id
        self.bos_labels = torch.ones((1, num_codebooks, 1)) * audio_encoder_bos_token_id
        self.num_samples = 0

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        self.num_samples += 1
        # Available keys in data:
        # 'dac', 'flac', 'length_ms', 'n_words', 'transcript', 'whisper_average_logprob',
        # 'whisper_max_logprob', 'whisper_min_logprob', 'whisper_sum_logprob'

        # Load audio (for reference embeddings)
        audio, sr = torchaudio.load(io.BytesIO(data["flac"]), format="flac")
        assert audio.size(0) == 1, f"Audio must be mono, but got {audio.size(0)} channels"

        if sr != self.audio_sr:
            resampler = torchaudio.transforms.Resample(sr, self.audio_sr)
            audio = resampler(audio)

        # Creating a reference audio segment of fixed length (audio_ref_len)
        audio = audio.squeeze()

        audio_len = audio.size(0)

        # Select a segment of audio to use
        if self.audio_ref_len and self.audio_ref_percentage:
            raise ValueError("Cannot specify both audio_ref_len and audio_ref_percentage")

        if not self.audio_ref_len and not self.audio_ref_percentage:
            raise ValueError("Must specify either audio_ref_len or audio_ref_percentage")

        if self.audio_ref_len is not None:
            # Check and pad if audio is shorter than required length
            if audio_len < self.audio_sr * self.audio_ref_len:
                pad = torch.zeros(self.audio_sr * self.audio_ref_len - audio.size(0))
                audio = torch.cat([audio, pad])

            if audio_len > self.audio_sr * self.audio_ref_len:
                start = torch.randint(0, audio_len - self.audio_sr * self.audio_ref_len, (1,)).item()
            else:
                start = 0  # Default to start at 0 if exactly the required length or error handled above

            audio = audio[start : start + self.audio_sr * self.audio_ref_len]

        elif self.audio_ref_percentage is not None:
            audio_ref_len = int(audio_len * self.audio_ref_percentage)
            start = torch.randint(0, audio_len - audio_ref_len, (1,)).item()  # (1,) - single random integer
            audio = audio[start : start + audio_ref_len]

        # Load DAC codes and re-arrange into the delay pattern
        labels = torch.tensor(data["dac"].astype(np.int64))
        assert (
            labels.size(0) == self.num_codebooks
        ), f"Number of codebooks != specified number of codebooks ({self.num_codebooks})"
        labels = labels.unsqueeze(0)
        # add bos
        labels = torch.cat([self.bos_labels, labels], dim=-1)

        labels, delay_pattern_mask = build_delay_pattern_mask(
            labels,
            bos_token_id=self.audio_encoder_bos_token_id,
            pad_token_id=self.audio_encoder_eos_token_id,
            max_length=labels.shape[-1] + self.num_codebooks,
            num_codebooks=self.num_codebooks,
        )

        # the first ids of the delay pattern mask are precisely labels, we use the rest of the labels mask
        # to take care of EOS
        # we want labels to look like this:
        #  - [B, a, b, E, E, E, E]
        #  - [B, B, c, d, E, E, E]
        #  - [B, B, B, e, f, E, E]
        #  - [B, B, B, B, g, h, E]
        labels = torch.where(delay_pattern_mask == -1, self.audio_encoder_eos_token_id, delay_pattern_mask)
        # the first timestamp is associated to a row full of BOS, let's get rid of it
        # we also remove the last timestampts (full of PAD)
        labels = labels[:, 1:]

        # Load and tokenize the transcription
        transcription = data["transcript"]
        transcript_tokens = self.prompt_tokenizer(transcription)["input_ids"]

        features = {
            "audio_ref": audio,
            "labels": labels,
            "prompt_input_ids": transcript_tokens,
            # "len_audio": len_audio # length of the original audio file, don't think we need this
        }

        return features


# class DataLoaderMDS(DataLoader)
class DataLoaderMDS(StreamingDataLoader):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: ParlerTTSTrainingArguments,
        manifest_path: str,
        batch_size: int,
        prompt_tokenizer: AutoTokenizer,
        audio_encoder_bos_token_id: int,
        audio_encoder_eos_token_id: int,
        shuffle: bool,
        collator: callable,  # TODO: Define this more specifically if possible
        drop_last: bool,
        epoch_size: Optional[int] = None,
        num_canonical_nodes: Optional[int] = None,
    ):
        configure_aws_creds()

        streams = gather_streams(
            manifest_path=manifest_path,
            s3_bucket_root=data_args.mds_s3_bucket_root,
            mds_cache_dir=data_args.mds_cache_dir,
        )

        dataset = DatasetMDS(
            streams=streams,
            batch_size=batch_size,
            prompt_tokenizer=prompt_tokenizer,
            num_codebooks=model_args.num_codebooks,
            audio_ref_sample_rate=model_args.audio_ref_encoder_sr,
            audio_ref_len=model_args.audio_ref_len,
            audio_ref_percentage=model_args.audio_ref_percentage,
            audio_encoder_bos_token_id=audio_encoder_bos_token_id,
            audio_encoder_eos_token_id=audio_encoder_eos_token_id,
            shuffle=shuffle,  # IterableDataset so we shuffle here rather than in DataLoader
            cache_limit=data_args.mds_cache_limit,
            epoch_size=epoch_size,
            num_canonical_nodes=num_canonical_nodes,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle is handled by the dataset if required
            collate_fn=collator,
            drop_last=drop_last,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
