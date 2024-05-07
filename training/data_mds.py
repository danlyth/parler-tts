from pathlib import Path
import io
from typing import Dict, List, Union, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from streaming import StreamingDataset
from streaming import Stream
from transformers import AutoTokenizer


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

    # Should now have e.g. ["s3://sesame-ml-datasets/libriheavy-small-3min-mds", "s3://sesame-ml-datasets/libriheavy-medium-3min-mds", etc.]
    streams = [Stream(remote=bucket, local=mds_cache_dir /f"{bucket.split('/')[-1]}") for bucket in buckets]

    return streams

class MDSSpeechDataset(StreamingDataset):
    def __init__(self,
                 streams: list, # list of Stream objects
                 batch_size: int,
                 prompt_tokenizer: AutoTokenizer,
                 audio_sample_rate: int,
                 audio_ref_len: int,
                 **kwargs):
        super().__init__(streams=streams,
                         batch_size=batch_size,
                         download_retry=kwargs['retry'],
                         shuffle=kwargs['shuffle'],
                         download_timeout=kwargs['timeout'],
                         cache_limit=kwargs['cache_limit'], # NOTE this isn't working for some reason...
                         )
        # self.shuffle = kwargs['shuffle'] # bool
        # self.prefetch = kwargs['prefetch'] # int, number of samples to prefetch
        # self.retry = kwargs['retry'] # int, number of retries for each sample
        # self.timeout = kwargs['timeout'] # int, timeout for each sample
        # self.batch_size = kwargs['batch_size'] # int
        # self.num_workers = num_workers # TODO, probably don't keep but might be useful for downloading
        self.sample_rate = kwargs['sample_rate']
        self.audio_sr = audio_sample_rate
        self.audio_ref_len = audio_ref_len
        self.prompt_tokenizer = prompt_tokenizer

        # self.channels = kwargs['channels']
        # self.segment_duration = kwargs['segment_duration']
        # self.total_frames = self.segment_duration * self.sample_rate
        # self.seek_time = 0.0

       
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Available keys in data:
        # 'dac', 'flac', 'length_ms', 'n_words', 'transcript', 'whisper_average_logprob',
        # 'whisper_max_logprob', 'whisper_min_logprob', 'whisper_sum_logprob'
        

        # flac file is stored as bytes, load appropriately
        audio, sr = torchaudio.load(io.BytesIO(data["flac"]), format="flac")
        # audio = data["flac"] # what sample rate?
        # sr = 16000 # TODO CHECK
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

@dataclass
class DataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        prompt_tokenizer (:class:`~transformers.AutoTokenizer`)
            The prompt_tokenizer used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    prompt_tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    prompt_max_length: Optional[int] = None
    audio_max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        
        # labels: DAC codes
        # prompt_input_ids: tokenized transcription
        # prompt_attention_mask: attention mask for tokenized transcription
        # audio_ref: reference audio segment (fixed length)
        # audio_ref_attention_mask: attention mask for reference audio segment
        # decoder_attention_mask: Optional attention mask for DAC codes if audio_max_length is not None and padding is "max_length"

        # Unlike default Parler-TTS we don't have "input_ids" (which they use for descriptions)

        # Transposing DAC codes so that we have (bsz, seq_len, num_codebooks)
        labels = [feature["labels"].transpose(0, 1) for feature in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # TODO remove this hardcoded value
        if self.audio_max_length is not None and self.padding == "max_length":
            labels = torch.nn.functional.pad(labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)))

        batch = {"labels": labels}

        # # TODO, need these variables - Don't think we do actually
        # output["len_audio"] = len_audio
        # # (1, bsz, codebooks, seq_len) -> (bsz, seq_len, codebooks)
        # output["labels"] = labels.squeeze(0).transpose(1, 2)
        # output["ratio"] = torch.ones_like(len_audio) * labels.shape[-1] / len_audio.max()

        prompt_input_ids = [{"input_ids": feature["prompt_input_ids"]} for feature in features]
        prompt_input_ids = self.prompt_tokenizer.pad(
            prompt_input_ids,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.prompt_max_length,
        )

        batch["prompt_input_ids"] = prompt_input_ids["input_ids"]
        if "attention_mask" in prompt_input_ids:
            batch["prompt_attention_mask"] = prompt_input_ids["attention_mask"]

        if self.audio_max_length is not None and self.padding == "max_length":
            # if we do torch.compile, we need to also specify the attention_mask
            # decoder_attention_mask = torch.ones(labels.shape[:2], dtype=input_ids["attention_mask"].dtype)
            decoder_attention_mask = torch.ones(labels.shape[:2], dtype=prompt_input_ids["attention_mask"].dtype)
            batch["decoder_attention_mask"] = decoder_attention_mask

        # These have a fixed length, no need to pad
        batch["audio_ref"] = torch.stack([feature["audio_ref"] for feature in features])
        # However, we would like an attention mask for the audio refs to help with compatibility with the model
        audio_ref_attention_mask = torch.ones((len(batch["audio_ref"]), batch["audio_ref"][0].shape[0]), dtype=torch.long)
        batch["audio_ref_attention_mask"] = audio_ref_attention_mask

        return batch