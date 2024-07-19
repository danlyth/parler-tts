import argparse
import time
import json

import torch
import torchaudio
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
)


def get_args():
    parser = argparse.ArgumentParser(description="Parler TTS Inference")
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the json file containing the arguments",
    )
    parser.add_argument(
        "--audio_ref_embedding_path",
        type=str,
        required=True,
        default=None,
        help="Path to the precomputed audio reference embeddings",
    )
    parser.add_argument(
        "--output_audio_path",
        type=str,
        required=True,
        default=None,
        help="Path to save the generated audio (include .wav extension)",
    )

    parser.add_argument(
        "--test_sentence",
        type=str,
        required=False,
        default="I guess if you've truly hit rock bottom, the only place is up.",
        help="Test sentence to generate audio for",
    )
    return parser.parse_args()


def main(
    json_path: str,
    audio_ref_embedding_path: str,
    output_audio_path: str,
    test_sentence: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = json.load(open(json_path))

    ############################################
    #         Load model and tokenizer         #
    ############################################

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

    model.to(device)
    model.eval()

    gen_kwargs = {
        "do_sample": model_args["do_sample"],
        "temperature": model_args["temperature"],
        "max_length": model_args["max_length"],
        "min_new_tokens": model_args["num_codebooks"] + 1,
    }

    ############################################
    #           Prepare inputs                 #
    ############################################

    encoder_outputs = torch.load(audio_ref_embedding_path).to(device)
    encoder_outputs = BaseModelOutput(encoder_outputs.unsqueeze(0))
    attention_mask = torch.ones((1, 1), dtype=torch.long).to(device)  # Encoder outputs is a single non-padded vector

    prompt = prompt_tokenizer(test_sentence, return_tensors="pt")
    prompt_input_ids = prompt["input_ids"].to(device)
    prompt_attention_mask = prompt["attention_mask"].to(device)

    # Pad prompt_input_ids and prompt_attention_mask to data_args.max_prompt_token_length, with leading zeros
    zero_padding = torch.zeros(
        (1, model_args["max_prompt_token_length"] - prompt_input_ids.shape[1]),
        dtype=torch.long,
    ).to(device)
    prompt_input_ids = torch.cat((zero_padding, prompt_input_ids), dim=1)
    prompt_attention_mask = torch.cat((zero_padding, prompt_attention_mask), dim=1)

    ############################################
    #               Generate                   #
    ############################################

    start_time = time.time()
    output_audios = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        **gen_kwargs,
    )
    end_time = time.time()
    print(f"Time taken for generation: {end_time - start_time}")
    torchaudio.save(
        output_audio_path,
        output_audios[0].cpu().unsqueeze(0),
        sample_rate=model_args["discrete_audio_feature_sample_rate"],
    )


if __name__ == "__main__":
    args = get_args()
    main(
        args.json_path,
        args.audio_ref_embedding_path,
        args.output_audio_path,
        args.test_sentence,
    )
