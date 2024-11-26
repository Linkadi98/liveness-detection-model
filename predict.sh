#!/bin/zsh

python3 test.py \
      --threshold 0.5 \
      --test_dir LCC_FASD/LCC_FASD_development \
      --epoch 20 \
      --model_file model/final_model.keras \
      --steps_per_epoch 260