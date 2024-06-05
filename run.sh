source ~/myenvs/p31_ptts/bin/activate

export $(grep -v '^#' .env | xargs)


rm -r /ephemeral_volume/ramon/tts/cache_parler-tts/ramontest

accelerate launch ./training/run_parler_tts_training_audio.py ./train_base_mls_ltr.json

