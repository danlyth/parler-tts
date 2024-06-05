import evaluate
import torch
from transformers import AutoModel, AutoProcessor, pipeline
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import numpy as np
from scipy.spatial.distance import cosine




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

def spk_sim(hyps, refs, device, per_device_eval_batch_size):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
    speaker_id_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

    speaker_id_model.to(device)
    speaker_id_model.eval()

    all_ref_embeds = []
    all_hyp_embeds = []

    with torch.no_grad():
        for i in range(0, len(refs), per_device_eval_batch_size):
            batch_ref = refs[i:i + per_device_eval_batch_size]
            batch_hyp = hyps[i:i + per_device_eval_batch_size]

            ref_features = feature_extractor(batch_ref, sampling_rate=16000, padding=True, return_tensors="pt").to(
                device)
            hyp_features = feature_extractor(batch_hyp, sampling_rate=16000, padding=True, return_tensors="pt").to(
                device)

            ref_embeds = speaker_id_model(**ref_features).embeddings
            hyp_embeds = speaker_id_model(**hyp_features).embeddings

            ref_embeds = torch.nn.functional.normalize(ref_embeds, dim=-1).cpu().numpy()
            hyp_embeds = torch.nn.functional.normalize(hyp_embeds, dim=-1).cpu().numpy()

            all_ref_embeds.extend(ref_embeds)
            all_hyp_embeds.extend(hyp_embeds)

    # Compute cosine similarity manually
    cosine_similarities = [1 - cosine(ref, hyp) for ref, hyp in zip(all_ref_embeds, all_hyp_embeds)]
    mean_cosine_similarity = np.mean(cosine_similarities)

    return mean_cosine_similarity, cosine_similarities


def wer(asr_model_name_or_path, prompts, audios, device, per_device_eval_batch_size, sampling_rate):
    metric = evaluate.load("wer")
    asr_pipeline = pipeline(model=asr_model_name_or_path, device=device)
    transcriptions = asr_pipeline(
        [{"raw": audio, "sampling_rate": sampling_rate} for audio in audios],
        batch_size=int(per_device_eval_batch_size),
    )

    word_error = 100 * metric.compute(
        predictions=[t["text"].lower() for t in transcriptions], references=[t.lower() for t in prompts]
    )

    return word_error, [t["text"] for t in transcriptions]


def compute_metrics(
    audios_hyp,
    audios_ref,
    prompts,
    asr_model_name_or_path,
    batch_size,
    prompt_tokenizer,
    sample_rate,
    device="cpu",
):
    results = {}

    prompts = prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)
    audios_hyp = [a.cpu().numpy() for a in audios_hyp]
    audios_ref = [a.cpu().numpy() for a in audios_ref]

    word_error, transcriptions = wer(asr_model_name_or_path, prompts, audios_hyp, device, batch_size, sample_rate)
    spk_similarity, speaker_similarities = spk_sim(audios_hyp, audios_ref, device, batch_size)

    results["wer"] = word_error
    results["speaker_sim"] = spk_similarity

    return results, prompts, audios_hyp, transcriptions
