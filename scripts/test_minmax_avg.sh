CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
nohup python -u test.py \
--spec test \
--save_prefix sa \
--exp_name minmax_avg \
--test_stride 0 \
--test_epoch -1 \
--save_res 2 \
>minmax_avg_test.txt 2>&1 &