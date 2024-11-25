#!/bin/zsh

#python3 dataset.py \
#      --input_dir LCC_FASD \
#      --output_dir data_resized \
#      --target_size 224

python3 train.py \
      --img_width 224 \
      --img_height 224 \
      --seed_number 24 \
      --train_batch_size 32 \
      --val_batch_size 32 \
      --num_epochs 20 \
      --input_data_path data_resized \
      --model_path model

#python3 tflite.py \
#      --saved_model_dir saved_model \
#      --tflite_file_path model.tflite