import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
from transformers import AutoTokenizer

from parler_tts.modeling_parler_tts import build_delay_pattern_mask


class DatasetLocal(torch.utils.data.Dataset):
    def __init__(
        self,
        root_audio_dir: str,  # directory containing audio files in train, dev, test subdirectories
        root_dac_dir: str,  # directory containing dac files in train, dev, test subdirectories
        metadata_path: str,  # path to tsv file with 2 columns: file_path, transcription
        prompt_tokenizer: AutoTokenizer,
        audio_sr: int = 16000,  # audio sample rate for reference encoder, typically 16kHz
        audio_ref_len: Optional[int] = None,  # audio reference length (in seconds) for reference encoder
        audio_ref_percentage: Optional[float] = None,  # percentage of audio to use for reference encoder
        num_codebooks: int = None,  # number of codebooks in the DAC features
        audio_encoder_bos_token_id: int = None,  # BOS token id for audio encoder
        audio_encoder_eos_token_id: int = None,  # EOS token id for audio encoder
        use_same_file_ref: bool = False,  # whether to use the same audio file as reference
        use_precomputed_ref_embed: bool = False,  # whether to use precomputed reference embeddings, one per style directory
        return_full_ref_audio: bool = False,  # whether to return the full reference audio file, useful for computing speaking similarity
    ):
        self.root_audio_dir = Path(root_audio_dir)
        self.root_dac_dir = Path(root_dac_dir)
        self.audio_sr = audio_sr
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
        self.use_same_file_ref = use_same_file_ref
        self.use_precomputed_ref_embed = use_precomputed_ref_embed
        self.return_full_ref_audio = return_full_ref_audio

        with open(metadata_path, "r") as f:
            self.data = f.readlines()

    def __getitem__(self, index):
        x = self.data[index]
        _, relative_audio_path, _, _, transcription = x.split("\t")
        # relative audio paths starts with "train/...", "dev/...", "test/...
        audio_path = self.root_audio_dir / relative_audio_path
        dac_path = (self.root_dac_dir / relative_audio_path).with_suffix(".pt")

        if not self.use_same_file_ref:
            # Use a different audio file from the same style directory as reference rather than the same audio file
            audio_path = random.choice(list(audio_path.parent.glob("*.wav")))

        audio, sr = torchaudio.load(audio_path)
        assert audio.size(0) == 1, f"Audio must be mono, but got {audio.size(0)} channels"

        if self.return_full_ref_audio:
            audio_ref_full = audio
            # TODO remove this hard-coding
            resampler = torchaudio.transforms.Resample(sr, 24000)  # 24kHz is the current sample rate of the DAC model
            audio_ref_full = resampler(audio_ref_full)
            audio_ref_full = audio_ref_full.squeeze()

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
        labels = torch.load(dac_path)  # Shape (n_codebooks, n_frames)
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

        # Tokenize the transcription
        transcript_tokens = self.prompt_tokenizer(transcription)["input_ids"]

        if self.use_precomputed_ref_embed:
            # Load precomputed reference embeddings

            #################################

            ref_embed_root = self.root_audio_dir.parent / "wavlm_base_plus_layer_5_mean_norm/"
            # TODO replace this hard-coding

            #################################

            relative_style_dir = Path(relative_audio_path).parent
            ref_embed_path = ref_embed_root / relative_style_dir / "mean_normalized.pt"
            ref_embed = torch.load(ref_embed_path)  # Size (1, embedding dimension)
            attention_mask = torch.ones(1)  # we're using mean embeddings, hence this shape

            if self.return_full_ref_audio:
                features = {
                    "audio_ref": audio,
                    "audio_ref_full": audio_ref_full,
                    "labels": labels,
                    "prompt_input_ids": transcript_tokens,
                    "encoder_outputs": ref_embed,
                    "attention_mask": attention_mask,
                }
            else:
                features = {
                    "audio_ref": audio,
                    "labels": labels,
                    "prompt_input_ids": transcript_tokens,
                    "encoder_outputs": ref_embed,
                    "attention_mask": attention_mask,
                }
        else:
            if self.return_full_ref_audio:
                features = {
                    "audio_ref": audio,
                    "audio_ref_full": audio_ref_full,
                    "labels": labels,
                    "prompt_input_ids": transcript_tokens,
                }
            else:
                features = {
                    "audio_ref": audio,
                    "labels": labels,
                    "prompt_input_ids": transcript_tokens,
                    # "len_audio": len_audio # length of the original audio file, don't think we need this
                }

        return features

    def __len__(self):
        return len(self.data)


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
        prompt_max_length (:obj:`int`, `optional`):
            The maximum length for the prompt sequences. If specified, will truncate sequences longer than this value.
        audio_max_length (:obj:`int`, `optional`):
            The maximum length for the audio sequences (measured in RVQ code frames). If specified, will truncate sequences longer than this value.
        ref_audio_max_length (:obj:`int`, `optional`):
            The maximum length for the reference audio sequences (measured in seconds). If specified, will truncate sequences longer than this value.
    """

    prompt_tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    prompt_max_length: Optional[int] = None
    audio_max_length: Optional[int] = None
    audio_ref_max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods

        # labels: DAC codes
        # prompt_input_ids: tokenized transcription
        # prompt_attention_mask: attention mask for tokenized transcription
        # audio_ref: reference audio segment (fixed length)
        # audio_ref_attention_mask: attention mask for reference audio segment
        # decoder_attention_mask: Optional attention mask for RVQ codes if audio_max_length is not None and padding is "max_length"

        # Unlike default Parler-TTS we don't have "input_ids" (which they use for descriptions)

        # Transposing DAC codes so that we have (bsz, seq_len, num_codebooks)
        pad_value = -100
        labels = [feature["labels"].transpose(0, 1) for feature in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_value)
        if self.audio_max_length is not None and self.padding == "max_length":
            # Crop to max length if required
            labels = labels[:, : self.audio_max_length, :]
            # Pad to max length if required
            labels = torch.nn.functional.pad(
                labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)), value=pad_value
            )

        batch = {"labels": labels}

        # # TODO, need these variables - Don't think we do actually
        # output["len_audio"] = len_audio
        # # (1, bsz, codebooks, seq_len) -> (bsz, seq_len, codebooks)
        # output["labels"] = labels.squeeze(0).transpose(1, 2)
        # output["ratio"] = torch.ones_like(len_audio) * labels.shape[-1] / len_audio.max()
        # TODO update from Yoach on this here: https://github.com/ylacombe/parler-tts/commit/0ce0df2caa3ee17fd21a38305ddd07fd263c1fb4#diff-e240f54d674b9dbab0d52426cb0abca99d46ec3e3d1b4000dd69d7e696034a3e

        # Crop prompt_input_ids to max length if required
        prompt_input_ids = [feature["prompt_input_ids"][: self.prompt_max_length] for feature in features]
        prompt_input_ids = [{"input_ids": x} for x in prompt_input_ids]
        # prompt_input_ids = [{"input_ids": feature["prompt_input_ids"]} for feature in features]
        # Crop prompt_input_ids to max length if required
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

        # the audio reference is not a fixed length, so we need to pad it and create the attention mask
        pad_value = -10000000
        audio_ref = [feature["audio_ref"] for feature in features]
        audio_ref = torch.nn.utils.rnn.pad_sequence(audio_ref, batch_first=True, padding_value=pad_value)
        if self.audio_ref_max_length is not None and self.padding == "max_length":
            # Crop to max length if required
            audio_ref = audio_ref[:, : self.audio_ref_max_length]
            # Pad to max length if required
            audio_ref = torch.nn.functional.pad(
                audio_ref, pad=(0, max(self.audio_ref_max_length - audio_ref.shape[1], 0)), value=pad_value
            )

        # Create attention mask with zeros where padding is applied
        audio_ref_attention_mask = torch.ones_like(audio_ref)
        audio_ref_attention_mask[audio_ref == pad_value] = 0
        batch["audio_ref"] = audio_ref
        batch["audio_ref_attention_mask"] = audio_ref_attention_mask

        if "encoder_outputs" in features[0]:  # True when fine-tuning with pre-computed embeddings
            encoder_outputs = [feature["encoder_outputs"] for feature in features]
            # These pre-computed embeddings are all the same size
            batch["encoder_outputs"] = torch.stack(encoder_outputs)
            attention_mask = [feature["attention_mask"] for feature in features]
            batch["attention_mask"] = torch.stack(attention_mask)

        else:
            # Downsample the audio attention mask frame-rate from 16kHz to 50Hz
            attention_mask = (
                torch.nn.functional.interpolate(
                    audio_ref_attention_mask.unsqueeze(1).float(), scale_factor=1 / 320, mode="nearest"
                )
                .squeeze(-1)
                .round()  # Ensure values are 0 or 1
                .int()  # Convert to integers (0 or 1)
            )
            batch["attention_mask"] = attention_mask.squeeze(1)
            # NOTE This needs to be padded/cropped to be exactly the same length as batch["encoder_outputs"]
            # but we'll do this in the training loop as we don't know the exact length here

        if "audio_ref_full" in features[0]:
            pad_value = 0  # In this case, we don't need an attention mask, so padding with 0 is fine
            audio_ref_full = [feature["audio_ref_full"] for feature in features]
            audio_ref_full = torch.nn.utils.rnn.pad_sequence(audio_ref_full, batch_first=True, padding_value=pad_value)
            batch["audio_ref_full"] = audio_ref_full

        return batch
