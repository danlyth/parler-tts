source ~/myenvs/p31_ptts/bin/activate

export $(grep -v '^#' .env | xargs)


rm -r /home/ramon/sesame/tts/parler-tts/exp/spk_sim_and_dacstream

accelerate launch ./training/run_parler_tts_training_audio.py ./train_base_mls_ltr.json

