#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo $CUDA_VISIBLE_DEVICES
models=("Qwen/Qwen2.5-Math-72B-Instruct" "Qwen/QVQ-72B-Preview" "meta-llama/Llama-3.3-70B-Instruct"
        "Qwen/Qwen2.5-32B-Instruct" "Qwen/QwQ-32B-Preview" "NovaSky-AI/Sky-T1-32B-Preview" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        "Qwen/Qwen2.5-14B-Instruct"
        "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct")

models=("Qwen/Qwen2.5-14B-Instruct")
# for model in "${models[@]}"
# do
#     echo "Running gpqa_diamond with model: $model}"
#     python gpqa_diamond.py --model "$model" | tee -a log.txt

#     echo "Running math500 with model: $model}"
#     python math500.py --model "$model" | tee -a log.txt

#     echo "Running aime2024 with model: $model}"
#     python aime2024.py --model "$model" | tee -a log.txt
# done

python print_result.py --task aime2024
python print_result.py --task gpqa_diamond
python print_result.py --task math500