import datetime
import json
import time
from pathlib import Path
from typing import List, Optional, Union

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from parler_tts import ParlerTTSForConditionalGeneration


# from training.eval import wer


# Available styles:
# conv-angry      conv-calm      conv-desire      conv-happy      conv-sad          read-confused    read-narration
# conv-animal     conv-child     conv-disgusted   conv-laughing   conv-sarcastic    read-default     read-sad
# conv-animaldir  conv-childdir  conv-enunciated  conv-narration  conv-sympathetic  read-enunciated  read-whisper
# conv-awe        conv-confused  conv-fast        conv-nonverbal  conv-whisper      read-happy
# conv-bored      conv-default   conv-fearful     conv-projected  read-laughing

MODEL_PATH = "/data/experiments/parler-tts/finetune_new/checkpoint-4000-epoch-111"
ROOT_REF_DIR = "/data/expresso/audio_48khz_short_chunks_ex02_processed/custom-ref"
# ROOT_REF_DIR = "/data/expresso/audio_48khz_short_chunks_ex02_processed/train"
# STYLE = ["conv-calm", "conv-default"]
# STYLE = "conv-default"
STYLE = "custom"
TEST_SENTENCES = "/data/expresso/audio_48khz_short_chunks_ex02_processed/test_text_only.txt"
# SINGLE_REF_FILE = "/data/expresso/audio_48khz_short_chunks_ex02_processed/train/conv-default/ex01-ex02_default_001-ex02_default_css000.wav"
SINGLE_REF_FILE = None
AUDIO_REF_LEN = None
AUDIO_SR = 16000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def generate(
    model_path: str,  # Path to the model checkpoint directory (not the model checkpoint itself)
    root_ref_dir: str,  # Directory containing reference audio files
    style: Union[str, List[str]],  # Style of the reference audio, e.g. "conv-calm" or ["conv-calm", "conv-default"]
    test_sentences: str,  # Path to the file containing test sentences
    single_ref_file: Optional[str] = None,  # (Optional) Path to a single reference audio file
    audio_ref_len: Optional[int] = None,  # (Optional) audio reference length (in seconds) for reference encoder
    audio_ref_sample_rate: int = 16000,  # Audio sample rate for reference encoder, typically 16kHz
    device: str = "cuda:0",
):
    root_ref_dir = Path(root_ref_dir)
    model_path = Path(model_path)

    # Load audio reference
    assert single_ref_file is not None or style is not None, "Either single_ref_file or style must be provided"
    audios = []

    if single_ref_file is not None:
        audio, sr = torchaudio.load(single_ref_file)
        audios.append(audio)

    else:
        if isinstance(style, list):
            for style in style:
                for audio_ref_path in (root_ref_dir / style).glob("*.wav"):
                    audio, sr = torchaudio.load(audio_ref_path)
                    audios.append(audio)

        else:
            for audio_ref_path in (root_ref_dir / style).glob("*.wav"):
                audio, sr = torchaudio.load(audio_ref_path)
                audios.append(audio)

    audio = torch.cat(audios, dim=1)
    audio = audio.squeeze()

    if sr != audio_ref_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, audio_ref_sample_rate)
        audio = resampler(audio)

    if audio_ref_len is not None:
        # Check and pad if audio is shorter than required length
        if audio.size(0) < audio_ref_sample_rate * audio_ref_len:
            pad = torch.zeros(audio_ref_sample_rate * audio_ref_len - audio.size(0))
            audio = torch.cat([audio, pad])

        # Check to ensure there's enough audio to select a segment
        if audio.size(0) > audio_ref_sample_rate * audio_ref_len:
            start = torch.randint(0, audio.size(0) - audio_ref_sample_rate * audio_ref_len, (1,)).item()
        else:
            start = 0  # Default to start at 0 if exactly the required length

        audio = audio[start : start + audio_ref_sample_rate * audio_ref_len]

    audio = audio.to(device)

    audio_ref_encoder = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
    audio_ref_encoder.to(device)

    with torch.no_grad():
        print("Encoding reference audio")
        encoder_outputs = audio_ref_encoder(audio.unsqueeze(0))

    # Load model
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    # Read sentences, export config and args
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(test_sentences, "r") as f:
        sentences = f.readlines()
        sentences = [sentence.strip() for sentence in sentences]

    output_dir = model_path / "outputs" / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_confg = model.generation_config.to_dict()
    with open(output_dir / f"{current_time}_generation_config.json", "w") as f:
        json.dump(generation_confg, f, indent=4)

    with open(output_dir / f"{current_time}_args.json", "w") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "root_ref_dir": str(root_ref_dir),
                "style": style,
                "test_sentences": test_sentences,
                "single_ref_file": single_ref_file,
                "audio_ref_len": audio_ref_len,
                "audio_ref_sample_rate": audio_ref_sample_rate,
                "device": device,
            },
            f,
            indent=4,
        )

    # Generate
    tik = time.time()
    for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Generating..."):
        prompt_input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
        generation = model.generate(encoder_outputs=encoder_outputs, prompt_input_ids=prompt_input_ids)
        if isinstance(style, list):
            output_path = output_dir / f"{current_time}_{style[0]}_{style[1]}_{i}.wav"
        else:
            output_path = output_dir / f"{current_time}_{style}_{i}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_path, audio_arr, model.config.sampling_rate)

    tok = time.time()
    print(
        f"Total time taken for {len(sentences)} sentences: {tok - tik} (average of {(tok - tik) / len(sentences)} per sentence)"
    )

    # word_error, transcriptions = wer(asr_model_name_or_path, sentences, audios, device, batch_size, sample_rate)


if __name__ == "__main__":
    generate(
        model_path=MODEL_PATH,
        root_ref_dir=ROOT_REF_DIR,
        style=STYLE,
        test_sentences=TEST_SENTENCES,
        single_ref_file=SINGLE_REF_FILE,
        audio_ref_len=AUDIO_REF_LEN,
        audio_ref_sample_rate=AUDIO_SR,
        device=DEVICE,
    )
