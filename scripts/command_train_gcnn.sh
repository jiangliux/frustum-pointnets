#/bin/bash
python3 train/train.py --gpu 1 --model frustum_pointnets_gcnn --log_dir train/log_gcnn_complete --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5