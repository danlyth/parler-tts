import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
import random
import json
import uuid
from dataclasses import dataclass
import re
from typing import Dict, Any, Tuple, List

import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
import whisper
from whisper.normalizers import EnglishTextNormalizer
import jiwer
import ray

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
)


def normalize_text(text: str) -> str:
    # Remove exclamation points
    text = re.sub(r"!", "", text)

    # Convert curly quotes to straight quotes
    text = re.sub(r"[“”]", '"', text)

    # Convert curly apostrophes to straight apostrophes
    text = re.sub(r"[‘’]", "'", text)

    return text


def ends_with_punct(s):
    """Filters out interruptions from the test sentences."""
    return len(s) > 0 and s[-1] in [".", "!", "?"]


@dataclass
class ParlerOptions:
    """Defines the arguments for generating audio from a sentence."""

    model_args: Dict[str, Any]
    audio_ref_embedding_path: str


class Parler:
    """Wrapper around the parler models to generate sentences."""

    def __init__(self, options: ParlerOptions):
        self._options = options
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = options.model_args
        self._model_args = options.model_args

        # Load the text tokenizer
        self._prompt_tokenizer = AutoTokenizer.from_pretrained(
            model_args["prompt_tokenizer_name"],
            cache_dir=None,
            use_fast=True,
            padding_side="left",
        )

        # Load the model config
        config = ParlerTTSConfig.from_pretrained(
            model_args["model_name_or_path"],
            cache_dir=None,
        )

        # Load the main model
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_args["model_name_or_path"],
            cache_dir=None,
            config=config,
        )

        model.to(self._device)
        model.eval()

        self._model = model
        self._audio_ref_embedding = torch.load(options.audio_ref_embedding_path).to(self._device)
        self._encoder_outputs = BaseModelOutput(self._audio_ref_embedding.unsqueeze(0))
        assert self._model_args["discrete_audio_feature_sample_rate"] == 24_000, f"unexpected"

    def generate(
        self,
        sentence: str,
    ) -> torch.Tensor:
        gen_kwargs = {
            "do_sample": self._model_args["do_sample"],
            "temperature": self._model_args["temperature"],
            "max_length": self._model_args["max_length"],
            "min_new_tokens": self._model_args["num_codebooks"] + 1,
        }

        attention_mask = torch.ones((1, 1), dtype=torch.long).to(
            self._device
        )  # Encoder outputs is a single non-padded vector

        prompt = self._prompt_tokenizer(sentence, return_tensors="pt")
        prompt_input_ids = prompt["input_ids"].to(self._device)
        prompt_attention_mask = prompt["attention_mask"].to(self._device)

        # Pad prompt_input_ids and prompt_attention_mask to data_args.max_prompt_token_length, with leading zeros
        zero_padding = torch.zeros(
            (
                1,
                self._model_args["max_prompt_token_length"] - prompt_input_ids.shape[1],
            ),
            dtype=torch.long,
        ).to(self._device)
        prompt_input_ids = torch.cat((zero_padding, prompt_input_ids), dim=1)
        prompt_attention_mask = torch.cat((zero_padding, prompt_attention_mask), dim=1)

        output_audios = self._model.generate(
            encoder_outputs=self._encoder_outputs,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            **gen_kwargs,
        )

        return output_audios[0].cpu().unsqueeze(0)


@ray.remote(num_gpus=0.2)
class ParlerRolloutActor:
    def __init__(self, options: ParlerOptions):
        self._parler = Parler(options)
        self._whisper_model = whisper.load_model("large-v3")
        self._resampler = torchaudio.transforms.Resample(
            orig_freq=24_000,
            new_freq=16_000,
        )
        self._objective_model = SQUIM_OBJECTIVE.get_model()

    @torch.inference_mode()
    def generate(
        self,
        sentence: str,
    ) -> Tuple[torch.Tensor, str, float, float, float]:
        """Returns: (audio tensor, whisper transcription)"""

        sentence = normalize_text(sentence)
        audio = self._parler.generate(sentence)
        resampled_audio = self._resampler(audio)
        transcription = self._whisper_model.transcribe(resampled_audio.squeeze(0))["text"]
        stoi_hyp, pesq_hyp, si_sdr_hyp = self._objective_model(audio)

        return (
            audio,
            transcription,
            stoi_hyp.item(),
            pesq_hyp.item(),
            si_sdr_hyp.item(),
        )


@dataclass
class Generation:
    text: str
    audio: torch.Tensor

    transcription: str
    stoi: float
    pesq: float
    si_sdr: float
    wer: float


@dataclass
class RolloutResults:
    """The results of a roll-out."""

    generations: List[Generation]
    wer: float
    avg_stoi: float
    avg_pesq: float
    avg_si_sdr: float


async def rollout(options: ParlerOptions, sentences: List[str], num_workers: int) -> RolloutResults:
    whisper_normalizer = EnglishTextNormalizer()
    sentences = sentences[:1000]  # NOTE: For testing purposes

    # Enqueue work.
    q = asyncio.Queue()
    for sentence in sentences:
        await q.put(sentence)

    outq = asyncio.Queue()

    async def queue_worker():
        actor = ParlerRolloutActor.remote(options)
        while True:
            try:
                sentence = await asyncio.wait_for(q.get(), timeout=1)
                results = await actor.generate.remote(sentence)
                await outq.put((sentence, results))
            except asyncio.TimeoutError:
                if q.empty():
                    break
                else:
                    await asyncio.sleep(0.1)

    workers = [asyncio.create_task(queue_worker()) for _ in range(num_workers)]

    async def finish_work():
        await asyncio.gather(*workers)
        await outq.put(None)

    asyncio.create_task(finish_work())

    idx = 0
    refs, hyps, stois, pesqs, si_sdrs = [], [], [], [], []
    generations = []
    while True:
        next_result = await outq.get()
        if next_result is None:
            break

        if idx % 1 == 0:
            print(f"Done with {idx}/{len(sentences)}")
        idx += 1
        sentence, (audio, transcript, stoi, pesq, si_sdr) = next_result

        ref_normalized = whisper_normalizer(sentence)
        hyp_normalized = whisper_normalizer(transcript)
        if ref_normalized.strip() == "" or hyp_normalized.strip() == "":
            continue

        individual_wer = jiwer.wer(
            reference=ref_normalized,
            hypothesis=hyp_normalized,
        )

        refs.append(ref_normalized)
        hyps.append(hyp_normalized)
        stois.append(stoi)
        pesqs.append(pesq)
        si_sdrs.append(si_sdr)

        generations.append(
            Generation(
                text=sentence,
                transcription=transcript,
                audio=audio,
                stoi=stoi,
                pesq=pesq,
                si_sdr=si_sdr,
                wer=individual_wer,
            )
        )

    wer = jiwer.wer(
        reference=refs,
        hypothesis=hyps,
    )

    return RolloutResults(
        generations=generations,
        wer=wer,
        avg_stoi=sum(stois) / len(stois),
        avg_pesq=sum(pesqs) / len(pesqs),
        avg_si_sdr=sum(si_sdrs) / len(si_sdrs),
    )


async def main(
    json_path: str,
    audio_ref_embedding_path: str,
    sentences_path: str,
    num_workers: int,
    ray_address: str = "",
):
    if ray_address != "":
        ray.init(
            address=ray_address,
            runtime_env={
                "working_dir": "",  # update this
                "excludes": ["outputs", "wandb"],
                "pip": [
                    "transformers>=4.39.0,<4.41.0",
                    "torch",
                    "sentencepiece",
                    "descript-audio-codec",
                ],
            },
        )

    model_args = json.load(open(json_path))
    sentences = json.load(open(sentences_path))

    options = ParlerOptions(
        model_args=model_args,
        audio_ref_embedding_path=audio_ref_embedding_path,
    )

    results = await rollout(options, sentences, num_workers)

    print(f"WER: {results.wer}")
    print(f"STOI: {results.avg_stoi}")
    print(f"PESQ: {results.avg_pesq}")
    print(f"SI-SDR: {results.avg_si_sdr}")
    output_dir = Path("")  # update this
    output_dir.mkdir(exist_ok=True, parents=True)
    for idx, generation in enumerate(results.generations):
        wer = generation.wer
        # convert wer to string with 3 significant digits
        wer = f"{wer:.3f}"
        output_path = output_dir / f"output_{idx}_wer_{wer}.wav"
        torchaudio.save(output_path, generation.audio, 24000)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
