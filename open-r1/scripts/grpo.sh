CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/sda/xjl/models/DeepSeek-R1-Distill-Qwen-1.5B

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/models/Qwen2.5-Math-1.5B
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/models/DeepSeek-R1-Distill-Qwen-1.5B

CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes 4 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml > terminal.txt

CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes 6 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml > terminal.txt

CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes 3 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml > terminal.txt


make evaluate MODEL=/mnt/xjl/checkpoints/sft-deepseek/DeepSeek-1.5B-GRPO-aime/checkpoint-877 TASK=aime24 PARALLEL=data NUM_GPUS=4