import argparse
import os
import time
from datetime import timedelta
from itertools import islice

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import AutoTokenizer

from training.data_local import DataCollator
from training.data_mds import DatasetMDS, configure_aws_creds, gather_streams


def parse_args():
    parser = argparse.ArgumentParser(description="Script for testing MDS dataloading.")

    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest file.")
    parser.add_argument("--s3_bucket_root", type=str, required=True, help="S3 bucket root.")
    parser.add_argument("--mds_cache_dir", type=str, required=True, help="MDS cache directory.")
    parser.add_argument("--retry", type=int, default=2, help="Number of retries for loading data.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for loading data.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker threads.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--cache_limit", type=str, default="10tb", help="Cache limit.")
    parser.add_argument("--num_codebooks", type=int, default=9, help="Number of codebooks.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to load from dataset.")
    parser.add_argument("--audio_ref_len", type=int, default=2, help="Length of audio reference.")
    parser.add_argument("--num_debug_samples", type=int, default=None, help="Number of samples to debug.")
    parser.add_argument("--max_audio_token_length", type=int, default=None, help="Max length of audio codes.")
    parser.add_argument("--max_prompt_token_length", type=int, default=None, help="Max length of prompt tokens.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument("--drop_last", action="store_true", help="Drop last incomplete batch.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory.")
    parser.add_argument("--use_accelerate", action="store_true", help="Prepare the dataloader with accelerate.")
    parser.add_argument("--debug", action="store_true", help="Breakpoint at dataset.")

    return parser.parse_args()


def main(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # load prompt tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base",
        cache_dir=None,
        token=None,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",  # prompt has to be padded on the left bc it's preprend to codebooks hidden states
    )

    if args.max_audio_token_length or args.max_prompt_token_length is not None:
        print(
            f"Using max audio token length: {args.max_audio_token_length} and max prompt token length: {args.max_prompt_token_length}"
        )
        data_collator = DataCollator(
            prompt_tokenizer,
            audio_max_length=args.max_audio_token_length,
            prompt_max_length=args.max_prompt_token_length,
            padding="max_length",
        )
    else:
        data_collator = DataCollator(prompt_tokenizer)

    configure_aws_creds()

    streams = gather_streams(args.manifest_path, args.s3_bucket_root, args.mds_cache_dir)

    dataset = DatasetMDS(
        streams,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        retry=args.retry,
        timeout=args.timeout,
        cache_limit=args.cache_limit,
        prompt_tokenizer=prompt_tokenizer,
        audio_ref_sample_rate=16000,
        audio_ref_len=args.audio_ref_len,
        epoch_size=args.num_samples,
        num_codebooks=args.num_codebooks,
    )

    print(f"Length of dataset is {len(dataset)}")

    if args.num_debug_samples is None:
        num_debug_samples = len(dataset)
    else:
        num_debug_samples = min(len(dataset), args.num_debug_samples)
    count = 0
    for i, features in tqdm(
        enumerate(islice(dataset, 0, num_debug_samples)), total=num_debug_samples, desc="Loading dataset"
    ):
        audio_ref = features["audio_ref"]
        labels = features["labels"]
        prompt_input_ids = features["prompt_input_ids"]
        count += 1
        if args.debug:
            print("Dataset loading correctly")
            print(f"Audio ref shape: {audio_ref.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Prompt input ids length: {len(prompt_input_ids)}")
            breakpoint()

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if args.use_accelerate:
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=60))]
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="bf16",
            log_with=["tensorboard"],
            project_dir="temp",
            dispatch_batches=False,  # TODO (Dan) testing this as our batches are not all the same length
            kwargs_handlers=kwargs_handlers,
        )
        print("Preparing dataloader with accelerate")
        dataloader = accelerator.prepare(dataloader)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # The 'length' of the dataloader is the length of the dataset/number of batches
    print(f"Length of dataloader is {len(dataloader)}")

    tic = time.time()
    count = 0
    codebook_shapes = []
    prompt_shapes = []
    for features in tqdm(dataloader):
        count += args.batch_size
        codebooks = features["labels"]
        codebook_shapes.append(codebooks.shape)
        prompt_input_ids = features["prompt_input_ids"]
        prompt_shapes.append(prompt_input_ids.shape)

        if count % (args.batch_size * 10) == 0:
            print(f"Count: {count}")
            print(f"Count per second: {count / (time.time() - tic):.2f}")
            if args.debug:
                breakpoint()

    toc = time.time()
    print(f"Finished loading in {toc - tic:.2f} seconds")
    print(f"Final count: {count}")
    print(f"Final count per second: {count / (toc - tic):.2f}")
    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
