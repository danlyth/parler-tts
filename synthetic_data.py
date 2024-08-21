from pathlib import Path
import json
import logging
import random
import sys
import time
import warnings

import torch
import torchaudio
from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
)

from transformers.modeling_outputs import BaseModelOutput

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
)
import librosa
from torchaudio.pipelines import SQUIM_OBJECTIVE
from accelerate import Accelerator
from jiwer import compute_measures


def generate(
    audio_ref_embedding_path: str,
    test_sentence: str,
    batch_size: int,
    device: torch.device,
    model: ParlerTTSForConditionalGeneration,
    model_args: dict,
    gen_kwargs: dict,
    prompt_tokenizer: AutoTokenizer,
    accelerator: Accelerator,
):
    encoder_outputs = torch.load(audio_ref_embedding_path).unsqueeze(0).to(device)
    attention_mask = torch.ones((1, 1), dtype=torch.long).to(device)  # Encoder outputs is a single non-padded vector
    encoder_outputs = BaseModelOutput(encoder_outputs)

    prompt = prompt_tokenizer(test_sentence, return_tensors="pt")
    prompt_input_ids = prompt["input_ids"].to(device)
    prompt_attention_mask = prompt["attention_mask"].to(device)

    # Pad prompt_input_ids and prompt_attention_mask to data_args.max_prompt_token_length, with leading zeros
    zero_padding = torch.zeros(
        (1, model_args["max_prompt_token_length"] - prompt_input_ids.shape[1]), dtype=torch.long
    ).to(device)
    prompt_input_ids = torch.cat((zero_padding, prompt_input_ids), dim=1)
    prompt_attention_mask = torch.cat((zero_padding, prompt_attention_mask), dim=1)

    batch = {}
    # Create batch_size copies of the encoder_outputs, attention_mask, prompt_input_ids, and prompt_attention_mask
    encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_outputs.last_hidden_state.repeat(batch_size, 1, 1),
        hidden_states=encoder_outputs.hidden_states,
    )
    attention_mask = attention_mask.repeat(batch_size, 1)
    prompt_input_ids = prompt_input_ids.repeat(batch_size, 1)
    prompt_attention_mask = prompt_attention_mask.repeat(batch_size, 1)
    batch["encoder_outputs"] = encoder_outputs
    batch["attention_mask"] = attention_mask
    batch["prompt_input_ids"] = prompt_input_ids
    batch["prompt_attention_mask"] = prompt_attention_mask

    generated_audios = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        **gen_kwargs,
    )

    return generated_audios


def transcribe(asr_pipeline, prompts, audios, batch_size, sampling_rate):
    return_language = None
    if isinstance(asr_pipeline.model, WhisperForConditionalGeneration):
        return_language = True

    transcriptions = asr_pipeline(
        [{"raw": audio, "sampling_rate": sampling_rate} for audio in audios],
        batch_size=int(batch_size),
        return_language=return_language,
    )
    # TODO - I was having trouble with this batch_size, might need to change it

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

    return normalized_predictions, normalized_references


def main(
    model_args_path: str,
    audio_ref_embedding_path: str,
    test_sentences_path: str,
    output_dir: str,
    batch_size: int,
    max_num_attempts: int,
    wer_threshold: float,
    cos_sim_threshold: float,
    pesq_threshold: float,
    stoi_threshold: float,
    si_sdr_threshold: float,
    shuffle: bool,
    shuffle_seed: int,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_precision = "bf16"
    # accelerator = Accelerator(mixed_precision=mixed_precision)
    accelerator = Accelerator()
    if not accelerator.is_main_process:
        warnings.filterwarnings("ignore")

    device = accelerator.device

    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    ################################
    #         Load models          #
    ################################

    model_args = json.load(open(model_args_path))

    # Load the text tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
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

    model = accelerator.prepare(model)
    # model = accelerator.unwrap_model(model, keep_fp32_wrapper=mixed_precision != "fp16")
    model = accelerator.unwrap_model(model)

    model.eval()

    gen_kwargs = {
        "do_sample": model_args["do_sample"],
        "temperature": model_args["temperature"],
        "max_length": model_args["max_length"],
        "min_new_tokens": model_args["num_codebooks"] + 1,
    }

    embed_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
    hidden_layer = 5
    embed_model.to(device)
    embed_model.eval()

    ref_emb = torch.load(audio_ref_embedding_path)
    ref_emb = ref_emb.to(device)

    objective_model = SQUIM_OBJECTIVE.get_model()
    objective_model.to(device)
    objective_model.eval()

    asr_model_name = "distil-whisper/distil-large-v3"

    asr_pipeline = pipeline(model=asr_model_name, device=device)

    ################################
    #      Generate candidate      #
    ################################

    test_sentences = json.load(open(test_sentences_path))
    indices = list(range(len(test_sentences)))

    if shuffle:
        # set seed so that the shuffle is the same across nodes
        random.seed(shuffle_seed)
        # want to keep the original indices of the sentences
        random.shuffle(indices)
        test_sentences = [test_sentences[i] for i in indices]

    for idx, test_sentence in zip(indices, test_sentences):
        overall_start_time = time.time()
        # Skip sentences if we already have a candidate generated in a previous run
        metadata_path = (output_dir / str(idx)).with_suffix(".json")
        if metadata_path.exists():
            logger.info(f"Skipping sentence {idx} as it already has a candidate.")
            continue

        logger.info(f"Generating candidates for sentence: {test_sentence} ({idx})")

        # We dynamically change the cosine threshold based on length of the sentence
        # This is because shorter sentences tend to have lower cosine similarity
        # TODO remove this hard-coding
        if len(test_sentence.split()) == 1:
            cos_sim_threshold = 0.70
        elif len(test_sentence.split()) > 1 and len(test_sentence.split()) <= 3:
            cos_sim_threshold = 0.725
        else:
            cos_sim_threshold = 0.75

        complete = False
        # see how many GPUs we're running on
        num_gpus = accelerator.num_processes
        num_attempts = max_num_attempts // num_gpus
        for attempt_id in range(num_attempts):
            logger.info(f"Attempt batch {attempt_id+1}")
            gen_start_time = time.time()
            output_audios = generate(
                audio_ref_embedding_path,
                test_sentence,
                batch_size,
                device,
                model,
                model_args,
                gen_kwargs,
                prompt_tokenizer,
                accelerator,
            )

            output_audios = accelerator.pad_across_processes((output_audios), dim=1, pad_index=0)
            output_audios = accelerator.gather_for_metrics((output_audios))

            gen_end_time = time.time()
            logger.info(f"Number of generations: {len(output_audios)}")
            logger.info(f"Time taken for generation: {gen_end_time - gen_start_time}")

            # if accelerator.is_main_process:  # only do evaluation on the main process TODO - is this necessary?
            eval_time_start = time.time()
            # If audio is CUDABFloat16Type, convert it to torch.cuda.FloatTensor
            # # TODO occurs when using multi-GPU, but not single GPU, investigate
            # output_audios = [audio.to(torch.float32) for audio in output_audios]

            # Evaluation model sample rate
            eval_model_sample_rate = 16000

            # Resample the audio to 16kHz
            resampler = torchaudio.transforms.Resample(
                model_args["discrete_audio_feature_sample_rate"], eval_model_sample_rate
            ).to(device)

            preds = [resampler(pred) for pred in output_audios]

            preds = [pred.cpu().numpy() for pred in output_audios]

            # Transcribe
            try:
                pred_texts, ref_texts = transcribe(
                    asr_pipeline,
                    [test_sentence] * len(preds),
                    preds,
                    batch_size,
                    eval_model_sample_rate,
                )
            except:  # TODO Need attention_mask for long generations in order to use batch_size > 1
                pred_texts, ref_texts = transcribe(
                    asr_pipeline,
                    [test_sentence] * len(preds),
                    preds,
                    1,
                    eval_model_sample_rate,
                )

            # logger.info(ref_texts)
            # logger.info(pred_texts)

            # Trim the audio to remove silence, this impacts embeddings, unfortunately have to do this on the cpu
            preds = [librosa.effects.trim(pred.cpu().numpy())[0] for pred in output_audios]

            # Move back on to the GPU for embeddings and audio fidelity metrics
            preds = [torch.from_numpy(pred).to(device) for pred in preds]

            # If any of the preds are too short, pad them
            min_len = 8000  # 0.5 seconds at 16kHz
            # Have to do this in a for loop because generations are not the same length.
            for i in range(len(preds)):
                if len(preds[i]) < min_len:
                    preds[i] = torch.nn.functional.pad(preds[i], (0, min_len - len(preds[i])))

            cos_sims = []
            stois = []
            pesqs = []
            si_sdrs = []
            wers = []
            with torch.no_grad():
                # Have to this in a for loop because generations are not the same length.
                for i in range(len(preds)):
                    # WavLM embeddings and cosine similarity
                    pred_emb = embed_model(preds[i].unsqueeze(0)).hidden_states[hidden_layer]
                    # Resulting embeddings are of shape (1, seq_len, hidden_size)
                    # Take the mean of the hidden states across the time dimension
                    pred_emb = torch.mean(pred_emb, dim=1)
                    cos_sim = torch.nn.functional.cosine_similarity(pred_emb, ref_emb, dim=-1)
                    cos_sims.append(cos_sim.cpu().item())

                    # Audio fidelity metrics
                    stoi, pesq, si_sdr = objective_model(preds[i].unsqueeze(0))
                    stois.append(stoi.cpu().item())
                    pesqs.append(pesq.cpu().item())
                    si_sdrs.append(si_sdr.cpu().item())

                    # WER
                    wer = compute_measures(ref_texts[i], pred_texts[i])["wer"]
                    wers.append(wer)
            # Check if any of the candidates pass all the thresholds
            for i in range(len(preds)):
                reject = False
                wer_pass = False
                cos_sim_pass = False

                if wers[i] > wer_threshold:
                    logger.info(f"{i} - WER above threshold: {wers[i]}")
                    reject = True
                else:
                    wer_pass = True
                if cos_sims[i] < cos_sim_threshold:
                    logger.info(f"{i} - Cosine similarity below threshold: {cos_sims[i]}")
                    reject = True
                else:
                    cos_sim_pass = True
                if wer_pass and cos_sim_pass:
                    logger.info("Only checking for audio fidelity; other checks passed.")

                if stois[i] < stoi_threshold:
                    logger.info(f"{i} - STOI below threshold: {stois[i]}")
                    reject = True
                if pesqs[i] < pesq_threshold:
                    logger.info(f"{i} - PESQ below threshold: {pesqs[i]}")
                    reject = True
                if si_sdrs[i] < si_sdr_threshold:
                    logger.info(f"{i} - SI-SDR below threshold: {si_sdrs[i]}")
                    reject = True
                if cos_sims[i] < cos_sim_threshold:
                    logger.info(f"{i} - Cosine similarity below threshold: {cos_sims[i]}")
                    reject = True

                if not reject:
                    # Save a json containing the transcript along with a sample of the normalized transcript and normalized predicted transcript
                    logger.info(
                        f"{i} -  Passed all thresholds (with cos sim of {round(cos_sims[i], 3)}, PESQ of {round(pesqs[i], 3)}, STOI of {round(stois[i], 3)}, SI-SDR of {round(si_sdrs[i], 3)}, and WER of {round(wers[i], 3)})"
                    )
                    complete = True
                    # Save the audio
                    output_audio_path = (Path(output_dir) / str(idx)).with_suffix(".wav")
                    torchaudio.save(
                        output_audio_path,
                        preds[i].cpu().unsqueeze(0),
                        sample_rate=model_args["discrete_audio_feature_sample_rate"],
                    )
                    # Save the metadata to a json file
                    metadata = {
                        "sentence": test_sentence,
                        "wer": wers[i],
                        "cos_sim": cos_sims[i],
                        "pesq": pesqs[i],
                        "stoi": stois[i],
                        "si_sdr": si_sdrs[i],
                        "audio_path": str(output_audio_path),
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                    break
            eval_time_end = time.time()
            logger.info(f"Time taken for evaluation: {eval_time_end - eval_time_start}")
            if complete:
                break
            overall_end_time = time.time()
            if not complete:
                logger.info(f"Failed to generate a candidate for sentence {idx} after {max_num_attempts} attempts.")
                # Check that one of the other processes has generated a candidate
                if metadata_path.exists():
                    logger.info(f"Another process has generated a candidate for sentence {idx}.")
                    break
                else:
                    # Save the metadata to a json file, containing all attempts
                    metadata = {
                        "sentence": test_sentence,
                        "norm_ref": ref_texts,
                        "norm_pred": pred_texts,
                        "wer": wers,
                        "cos_sim": cos_sims,
                        "pesq": pesqs,
                        "stoi": stois,
                        "si_sdr": si_sdrs,
                    }
                    metadata_failed_path = (output_dir / "failed" / str(idx)).with_suffix(".json")
                    metadata_failed_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(metadata_failed_path, "w") as f:
                        json.dump(metadata, f, indent=4)

            else:
                logger.info(f"Successfully generated a candidate for sentence {idx}.")
            logger.info(f"Total time taken for sentence {idx}: {overall_end_time - overall_start_time}")


if __name__ == "__main__":
    main(
        model_args_path="/shared/production_tts/jordana_tts_args_v3.json",
        audio_ref_embedding_path="/shared/production_tts/wavlm_layer_5_mean_embedding_cos_sim_08.pt",
        test_sentences_path="/shared/sesame_voice_recordings/recording_session_2024_06/synthetic_data/companion_utts.no_end_punc.json",
        output_dir="/shared/sesame_voice_recordings/recording_session_2024_06/synthetic_data/synthetic_data_2024_08_14_no_end_punc",
        batch_size=8,  # was 16 but I don't think we need that many
        max_num_attempts=8,
        wer_threshold=0.0,
        cos_sim_threshold=0.75,
        pesq_threshold=3.3,
        stoi_threshold=0.98,
        si_sdr_threshold=20.0,
        shuffle=True,  # Note, setting to true doesn't work on multi-GPU setup
        shuffle_seed=1,
    )
