from typing import List, Tuple
import evaluate
from jiwer import compute_measures
import librosa
import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
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

    measures = compute_measures(normalized_references, normalized_predictions)
    word_error = 100 * measures["wer"]
    substitutions = measures["substitutions"]
    deletions = measures["deletions"]
    insertions = measures["insertions"]
    # word_error = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)

    return word_error, substitutions, deletions, insertions, [t["text"] for t in transcriptions]


def spk_sim(
    preds: List[torch.Tensor],
    refs: List[torch.Tensor],
    device: torch.device,
    model_type: str = None,
) -> Tuple[float, torch.Tensor]:
    if model_type == "speaker_verification":
        embed_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    elif model_type == "embedding":
        embed_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
        hidden_layer = 5
    else:
        raise ValueError(f"Model {model_type} not supported for speaker similarity")
    embed_model.to(device)
    embed_model.eval()

    # If any of the refs or preds are too short, pad them
    min_len = 8000  # 0.5 seconds at 16kHz
    for i in range(len(preds)):
        if len(preds[i]) < min_len:
            preds[i] = torch.nn.functional.pad(preds[i], (0, min_len - len(preds[i])))
    for i in range(len(refs)):
        if len(refs[i]) < min_len:
            refs[i] = torch.nn.functional.pad(refs[i], (0, min_len - len(refs[i])))

    # Have to this in a for loop because generations are not the same length.
    pred_embs = []
    ref_embs = []
    with torch.no_grad():
        for pred, ref in zip(preds, refs):
            if model_type == "speaker_verification":
                pred_emb = embed_model(pred.unsqueeze(0)).embeddings
                ref_emb = embed_model(ref.unsqueeze(0)).embeddings
            elif model_type == "embedding":
                pred_emb = embed_model(pred.unsqueeze(0)).hidden_states[hidden_layer]
                ref_emb = embed_model(ref.unsqueeze(0)).hidden_states[hidden_layer]
                # Resulting embeddings are of shape (1, seq_len, hidden_size)
                # Take the mean of the hidden states across the time dimension
                pred_emb = torch.mean(pred_emb, dim=1)
                ref_emb = torch.mean(ref_emb, dim=1)
            pred_embs.append(pred_emb)
            ref_embs.append(ref_emb)

    pred_embs = torch.stack(pred_embs, dim=0)
    ref_embs = torch.stack(ref_embs, dim=0)

    cosine_similarities = torch.nn.functional.cosine_similarity(pred_embs, ref_embs, dim=-1).cpu()
    mean_cosine_similarity = torch.mean(cosine_similarities).cpu().item()

    return mean_cosine_similarity, cosine_similarities


def audio_fidelity(
    preds: List[torch.Tensor],
    refs: List[torch.Tensor],
    device: torch.device,
) -> Tuple[float, float, float, float]:
    objective_model = SQUIM_OBJECTIVE.get_model()
    objective_model.to(device)
    objective_model.eval()

    # Have to this in a for loop because generations are not the same length.
    with torch.no_grad():
        gen_stoi_scores = []
        gen_pesq_scores = []
        gen_si_sdr_scores = []
        for pred in preds:
            stoi, pesq, si_sdr = objective_model(pred.unsqueeze(0))
            gen_stoi_scores.append(stoi)
            gen_pesq_scores.append(pesq)
            gen_si_sdr_scores.append(si_sdr)
        gen_stoi_scores = torch.stack(gen_stoi_scores, dim=0)
        gen_pesq_scores = torch.stack(gen_pesq_scores, dim=0)
        gen_si_sdr_scores = torch.stack(gen_si_sdr_scores, dim=0)

        gen_mean_stoi = torch.mean(gen_stoi_scores).cpu().item()
        gen_mean_pesq = torch.mean(gen_pesq_scores).cpu().item()
        gen_mean_si_sdr = torch.mean(gen_si_sdr_scores).cpu().item()

        ref_pesq_scores = []
        for ref in refs:
            stoi, pesq, si_sdr = objective_model(ref.unsqueeze(0))
            ref_pesq_scores.append(pesq)
        ref_pesq_scores = torch.stack(ref_pesq_scores, dim=0)
        ref_mean_pesq = torch.mean(ref_pesq_scores).cpu().item()

    return gen_mean_stoi, gen_mean_pesq, gen_mean_si_sdr, ref_mean_pesq


def compute_metrics(
    audio_preds,
    audio_refs,
    prompts,
    asr_model_name_or_path,
    batch_size,
    prompt_tokenizer,
    gen_model_sample_rate,
    audio_ref_encoder_sr,  # Both WavLM and Whisper require 16kHz audio, so OK to use one param for both for now
    device="cpu",
):
    results = {}

    prompts = prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)

    # Need to ensure that audio_preds have the same sampling rate as the ASR model and speaker similarity model
    # audio_refs are already at 16kHz so no need to resample
    resampler = torchaudio.transforms.Resample(gen_model_sample_rate, audio_ref_encoder_sr).to(device)
    audio_preds = [resampler(audio) for audio in audio_preds]

    # Currently audio_refs are padded with -10000000, need to replace this padding with zeros
    # TODO - should we really be padding with -10000000 in the first place?
    audio_refs = [ref * (ref != -10000000) for ref in audio_refs]

    # Trim trailing and leading silence to ensure that the speaker/embedding similarity are not affected by batch padding
    # Unfortunately, we have to do this on Numpy arrays and unfortunately this means they are no longer the same length
    audio_preds = [librosa.effects.trim(pred.cpu().numpy())[0] for pred in audio_preds]
    audio_refs = [librosa.effects.trim(ref.cpu().numpy())[0] for ref in audio_refs]

    # We've already re-sampled to audio_ref_encoder_sr
    # This Whisper pipeline expects audio in Numpy arrays
    word_error, substitutions, deletions, insertions, transcriptions = wer(
        asr_model_name_or_path, prompts, audio_preds, device, 1, audio_ref_encoder_sr
    )
    # Using batch size of 1 now that audio files are different lengths

    # Now put the audio back on the device for the other models
    audio_preds = [torch.from_numpy(pred).to(device) for pred in audio_preds]
    audio_refs = [torch.from_numpy(ref).to(device) for ref in audio_refs]

    spk_similarity, _ = spk_sim(audio_preds, audio_refs, device, model_type="speaker_verification")
    emb_similarity, _ = spk_sim(audio_preds, audio_refs, device, model_type="embedding")
    gen_stoi, gen_pesq, gen_si_sdr, ref_pesq = audio_fidelity(audio_preds, audio_refs, device)

    results["speaker_sim"] = spk_similarity
    results["embedding_sim"] = emb_similarity
    results["pesq_gen"] = gen_pesq
    results["stoi_gen"] = gen_stoi
    results["si_sdr_gen"] = gen_si_sdr
    results["pesq_ref"] = ref_pesq
    results["wer"] = word_error
    results["substitutions"] = substitutions
    results["deletions"] = deletions
    results["insertions"] = insertions

    return results, prompts, transcriptions
