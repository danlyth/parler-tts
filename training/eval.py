from typing import List, Tuple
import evaluate
import librosa
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, pipeline
from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline,
    WavLMForXVector,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
)


def clap_similarity(clap_model_name_or_path, texts, audios, device):
    clap = AutoModel.from_pretrained(clap_model_name_or_path)
    clap_processor = AutoProcessor.from_pretrained(clap_model_name_or_path)
    clap_inputs = clap_processor(text=texts, audios=audios, padding=True, return_tensors="pt").to(device)
    clap.to(device)
    with torch.no_grad():
        text_features = clap.get_text_features(
            clap_inputs["input_ids"], attention_mask=clap_inputs.get("attention_mask", None)
        )
        audio_features = clap.get_audio_features(clap_inputs["input_features"])

        cosine_sim = torch.nn.functional.cosine_similarity(audio_features, text_features, dim=1, eps=1e-8)

    clap.to("cpu")
    clap_inputs.to("cpu")
    return cosine_sim.mean().to("cpu")


def wer(asr_model_name_or_path, prompts, audios, device, per_device_eval_batch_size, sampling_rate):
    metric = evaluate.load("wer")
    asr_pipeline = pipeline(model=asr_model_name_or_path, device=device)

    return_language = None
    if isinstance(asr_pipeline.model, WhisperForConditionalGeneration):
        return_language = True

    transcriptions = asr_pipeline(
        [{"raw": audio, "sampling_rate": sampling_rate} for audio in audios],
        batch_size=int(per_device_eval_batch_size),
        return_language=return_language,
    )
    # TODO - I was having trouble with this per_device_eval_batch_size, might need to change it

    if isinstance(asr_pipeline.tokenizer, (WhisperTokenizer, WhisperTokenizerFast)):
        tokenizer = asr_pipeline.tokenizer
    else:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    english_normalizer = tokenizer.normalize
    basic_normalizer = tokenizer.basic_normalize

    normalized_predictions = []
    normalized_references = []

    for pred, ref in zip(transcriptions, prompts):
        normalizer = (
            english_normalizer if return_language and pred["chunks"][0]["language"] == "english" else basic_normalizer
        )
        norm_ref = normalizer(ref)
        if len(norm_ref) > 0:
            norm_pred = normalizer(pred["text"])
            normalized_predictions.append(norm_pred)
            normalized_references.append(norm_ref)

    word_error = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)

    return word_error, [t["text"] for t in transcriptions]


def spk_sim(
    preds: List[torch.Tensor],
    refs: List[torch.Tensor],
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    speaker_id_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    speaker_id_model.to(device)
    speaker_id_model.eval()

    # Need to trim silence to get decent results and unfortunately this needs to be done in Librosa on the CPU
    preds = [librosa.effects.trim(pred.cpu().numpy())[0] for pred in preds]
    refs = [librosa.effects.trim(ref.cpu().numpy())[0] for ref in refs]  # TODO don't need to do this for references

    # Then need to put them back on to the device
    preds = [torch.from_numpy(pred).to(device) for pred in preds]
    refs = [torch.from_numpy(ref).to(device) for ref in refs]

    # If any of the refs or preds are too short, pad them
    min_len = 8000  # 0.5 seconds at 16kHz
    for i in range(len(preds)):
        if len(preds[i]) < min_len:
            preds[i] = torch.nn.functional.pad(preds[i], (0, min_len - len(preds[i])))
    for i in range(len(refs)):
        if len(refs[i]) < min_len:
            refs[i] = torch.nn.functional.pad(refs[i], (0, min_len - len(refs[i])))

    pred_embs = []
    ref_embs = []
    with torch.no_grad():
        for pred, ref in zip(preds, refs):
            pred_emb = speaker_id_model(pred.unsqueeze(0)).embeddings
            ref_emb = speaker_id_model(ref.unsqueeze(0)).embeddings
            pred_embs.append(pred_emb)
            ref_embs.append(ref_emb)

    pred_embs = torch.stack(pred_embs, dim=0)
    ref_embs = torch.stack(ref_embs, dim=0)

    cosine_similarities = torch.nn.functional.cosine_similarity(pred_embs, ref_embs, dim=-1).cpu()
    mean_cosine_similarity = torch.mean(cosine_similarities).cpu().item()

    return mean_cosine_similarity, cosine_similarities


def compute_metrics(
    audio_preds,
    audio_refs,
    prompts,
    asr_model_name_or_path,
    batch_size,
    prompt_tokenizer,
    sample_rate,
    device="cpu",
):
    results = {}

    prompts = prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)

    # Need to ensure that audio_preds have the same sampling rate as the ASR model and speaker similarity model
    resampler = torchaudio.transforms.Resample(sample_rate, 16_000).to(device)  # TODO, remove this hard-coding
    audio_preds_resampled = [resampler(audio) for audio in audio_preds]

    spk_similarity, _ = spk_sim(audio_preds_resampled, audio_refs, device)
    audio_preds_np = [a.cpu().numpy() for a in audio_preds]  # Required for ASR pipeline
    audio_preds_np = [librosa.effects.trim(pred)[0] for pred in audio_preds_np]  # Removing trailing silence
    # word_error, transcriptions = wer(asr_model_name_or_path, prompts, audio_preds_np, device, batch_size, sample_rate)
    word_error, transcriptions = wer(asr_model_name_or_path, prompts, audio_preds_np, device, 1, sample_rate)

    results["speaker_sim"] = spk_similarity
    results["wer"] = word_error

    return results, prompts, audio_preds, transcriptions
