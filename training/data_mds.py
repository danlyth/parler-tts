from pathlib import Path
import os
import io

import numpy as np
import torch
import torchaudio
from streaming import StreamingDataset
from streaming import Stream
from transformers import AutoTokenizer


def configure_aws_creds():
    os.environ["S3_ENDPOINT_URL"] = f"https://{os.environ['CF_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CF_AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CF_AWS_SECRET_ACCESS_KEY"]


def gather_streams(manifest_path: str,
                   s3_bucket_root: str,
                   mds_cache_dir: str):
    """Instantiate SpeechDataset from a path to a directory containing a manifest as a txt file.

    Args:
        manifest_path (str or Path): Path to manifest text file containing list of dataset names.
        s3_bucket_root (str): Path to s3 bucket containing the above datasets.
        mds_cache_dir (str or Path): Path to local directory to store the downloaded datasets.
    """
    mds_cache_dir = Path(mds_cache_dir)

    with open(manifest_path, "r") as f:  # type: ignore
        dataset_names = f.read().splitlines()
    dataset_names = [dataset_name.strip() for dataset_name in dataset_names]

    # s3_bucket_root is, for example, s3://my-data-bucket/
    buckets = [str(s3_bucket_root + dataset_name) for dataset_name in dataset_names]
    assert len(buckets) > 0, "No datasets found in manifest."

    # Should now have e.g. ["s3://my-data-bucket/mls_eng_train_dac", "s3://my-data-bucket/mls_eng_dev_dac", etc.]
    streams = [Stream(remote=bucket, local=mds_cache_dir /f"{bucket.split('/')[-1]}") for bucket in buckets]

    return streams


class DatasetMDS(StreamingDataset):
    def __init__(self,
                 streams: list, 
                 batch_size: int, 
                 prompt_tokenizer: AutoTokenizer,
                 audio_sample_rate: int,
                 audio_ref_len: int,
                 **kwargs):
        super().__init__(streams=streams, # list of Stream objects
                         batch_size=batch_size, # necessary for deterministic resumption and optimal performance
                        #  download_retry=kwargs["retry"], # Default is 2
                        #  download_timeout=kwargs["timeout"], # Default is 60
                         shuffle=kwargs["shuffle"], # Default is False
                         cache_limit=kwargs["cache_limit"], # NOTE this isn't working for some reason...
                         # See https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html
                        #  for full list of available arguments
                         )
        
        self.audio_sr = audio_sample_rate
        self.audio_ref_len = audio_ref_len
        self.prompt_tokenizer = prompt_tokenizer

       
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

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

        # Check and pad if audio is shorter than required length
        if audio.size(0) < self.audio_sr * self.audio_ref_len:
            pad = torch.zeros(self.audio_sr * self.audio_ref_len - audio.size(0))
            audio = torch.cat([audio, pad])

        # Check to ensure there's enough audio to select a segment
        if audio.size(0) > self.audio_sr * self.audio_ref_len:
            start = torch.randint(0, audio.size(0) - self.audio_sr * self.audio_ref_len, (1,)).item()
        else:
            start = 0  # Default to start at 0 if exactly the required length or error handled above

        audio = audio[start:start + self.audio_sr * self.audio_ref_len]

        # Load DAC codes
        # labels = torch.from_numpy(data["dac"])
        # With the above I'm getting this error, going to try this instead TypeError: can't convert np.ndarray of type numpy.uint16. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        labels = np.frombuffer(data["dac"], dtype=np.uint16)
        labels = torch.from_numpy(labels.astype(np.int32)).unsqueeze(0) # Adding channel dimension


        # Load transcription
        transcription = data["transcript"]
        # Tokenize the transcription
        transcript_tokens = self.prompt_tokenizer(transcription)["input_ids"]

        features = {"audio_ref": audio,
                    "labels": labels,
                    "prompt_input_ids": transcript_tokens,
                    # "len_audio": len_audio # length of the original audio file, don't think we need this
                    } 

        return features