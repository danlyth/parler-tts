import argparse
from pathlib import Path

import torch
import torchaudio
from torch import nn

from parler_tts import DACConfig, DACModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--audio_test_path", type=str, required=True)
    parser.add_argument("--audio_output_path", type=str, required=True)
    parser.add_argument("--test_encoding_decoding", action="store_true")
    return parser.parse_args()


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(Path(args.pretrained_model_dir) / "weights.pth")
    metadata = state_dict["metadata"]
    state_dict = state_dict["state_dict"]
    logs = torch.load(Path(args.pretrained_model_dir) / "metadata.pth")["logs"]
    val_l1_loss = logs["val"]["waveform/loss"][-1]  # Used to check if the model is working correctly

    print("Initializing DAC model with the following configuration:")
    for key, value in metadata["kwargs"].items():
        print(f"{key}: {value}")

    num_codebooks = metadata["kwargs"]["n_codebooks"]
    codebook_size = metadata["kwargs"]["codebook_size"]
    latent_dim = metadata["kwargs"]["latent_dim"]
    sampling_rate = metadata["kwargs"]["sample_rate"]
    encoder_rates = metadata["kwargs"]["encoder_rates"]

    # The following are unused but we'll keep them here for reference:
    # encoder_dim = metadata["kwargs"]["encoder_dim"]
    # decoder_dim = metadata["kwargs"]["decoder_dim"]
    # decoder_rates = metadata["kwargs"]["decoder_rates"]
    # codebook_dim = metadata["kwargs"]["codebook_dim"]
    # quantizer_dropout = metadata["kwargs"]["quantizer_dropout"]

    frame_rate = sampling_rate
    for encoder_rate in encoder_rates:
        frame_rate = frame_rate / encoder_rate

    # Take log 2 of the codebook size to get the number of bits per codebook
    bits_per_codebook = torch.log2(torch.tensor(codebook_size)).item()
    model_bitrate = bits_per_codebook * num_codebooks * frame_rate / 1000  # in kbps, hence the division by 1000
    print(f"Model bitrate: {model_bitrate:.2f} kbps")
    print(f"Frame rate: {frame_rate:.2f} Hz")

    config = DACConfig(
        num_codebooks=num_codebooks,
        model_bitrate=model_bitrate,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        frame_rate=frame_rate,
        sampling_rate=sampling_rate,
    )

    model = DACModel(config)

    # The DACModel appends "model" to all its keys, the state_dict does not
    state_dict = {f"model.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)

    if args.test_encoding_decoding:
        audio, sr = torchaudio.load(args.audio_test_path)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            audio = resampler(audio)

        audio = audio.to(device)

        with torch.no_grad():
            encoded = model.encode(audio.unsqueeze(0))
            codes = encoded[0]
            audio_scales = encoded[1]

            decoded = model.decode(codes, audio_scales)
            decoded_audio = decoded[0].squeeze(1)

        # Check that original audio and decoded audio are approximately the same length (within 500 samples - ~20ms at 24kHz)
        assert (
            abs(audio.size(-1) - decoded_audio.size(-1)) <= 500
        ), "Original and decoded audio are not approximately the same length"

        if audio.size(-1) < decoded_audio.size(-1):
            decoded_audio = decoded_audio[:, : audio.size(-1)]

        if audio.size(-1) > decoded_audio.size(-1):
            audio = audio[:, : decoded_audio.size(-1)]

        # Check that the L1 loss between the original and decoded audio is approximately the same as the validation L1 loss
        l1_loss = nn.L1Loss()(audio, decoded_audio).item()

        assert (
            abs(l1_loss - val_l1_loss) < 0.1
        ), "L1 loss between original and decoded audio is not approximately the same as the validation L1 loss"

        torchaudio.save(args.audio_output_path, decoded_audio.cpu(), sampling_rate)

    model.save_pretrained(args.output_model_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
