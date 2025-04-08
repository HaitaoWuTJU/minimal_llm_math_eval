# Minimal LLM Math Eval

Zero-shot CoT
```
# math500
python math500.py --model Qwen/Qwen2.5-72B-Instruct

# gpqa diamond
python gpqa_diamond.py --model Qwen/Qwen2.5-72B-Instruct

# aime2024
python aime2024.py --model Qwen/Qwen2.5-72B-Instruct
```

Echo result
```
python print_result.py --task aime2024
python print_result.py --task gpqa_diamond
python print_result.py --task math500
```