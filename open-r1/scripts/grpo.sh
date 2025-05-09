iCUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/sda/xjl/models/DeepSeek-R1-Distill-Qwen-1.5B




CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml