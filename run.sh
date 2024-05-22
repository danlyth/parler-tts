source ~/myenvs/p31_ptts/bin/activate

export $(grep -v '^#' .env | xargs)


accelerate launch ./training/run_parler_tts_training_audio.py ./train_base_mls_ltr.json

