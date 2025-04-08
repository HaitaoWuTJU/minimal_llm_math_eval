import json
from tqdm import tqdm
import argparse
import glob

seed = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with main.py")
    parser.add_argument("--task", type=str, default='math500', help="Benchmark")
    args = parser.parse_args()

    task = args.task
    files= glob.glob(f"./results/{task}/*_cot_chat.json")

    print(f"\n------task: {task}------\n")
    for file in files:
        model_name = file.rsplit('/',1)[-1].split('_cot_chat.json',1)[0]
        with open(file, 'r') as f:
            save_data = json.load(f)

        correct = 0
        total = len(save_data)
        for i in range(len(save_data)):
            i = str(i)
            sample = save_data[i]
            
            response =  sample['ans']
            gold = sample['gold']
            pred = sample['pred']
            acc = sample['acc']
            correct += acc

        accuracy = correct/len(save_data)
        print(f"Model: {model_name}, Accuracy: {accuracy:.2%} {correct=} {total=}")
