CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
--spec train \
--dataset_prefix HC_Anomaly_6-12s_withSP_strict \
--data_split 0 \
--data_dir ../PULSE_data \
--aug none \
--ds 0.5 \
--seq_len 10 \
--labels hc csp lv \
--arch blo \
--gdl 1 \
--gdl_weight 0.0043 0.6875 0.3082 \
--start_epoch 0 \
--epochs 3000 \
--save_prefix sa \
--exp_name minmax_avg \
--batch_size 16 \
--workers 4 \
--save_freq 50 \
--verbose_freq 2 \
--lower_loss dsc \
--lower_lr 1e-4 \
--upper_loss mse \
--upper_lr 1e-4 \
--score avg \
--score_norm 0-1 \
>minmax_avg.txt 2>&1 &