import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
import argparse
from transformers import set_seed
from poex.models.openai_model import OpenaiModel
from poex.datasets.jailbreak_datasets import JailbreakDataset
from poex.models.huggingface_model import from_pretrained
from poex.attacker.POEX import POEX
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
set_seed(42)

def main(args):

    # Load dataset
    dataset = JailbreakDataset(args.dataset_path)

    # Load attack model
    ATTACK_MODEL = from_pretrained(args.attack_model_path, args.attack_model_name, max_new_tokens=512, mode='attack')
    
    # Load target model
    TARGET_MODEL = ATTACK_MODEL
    
    # Load unaligned model
    UNALIGNED_MODEL = from_pretrained(args.attack_model_path, args.attack_model_name, max_new_tokens=512, mode='unaligned')
    
    # Load eval model
    eval_model_name = 'gpt-4o'
    api_key = ''
    EVAL_MODEL = OpenaiModel(model_name=eval_model_name, api_keys=api_key, base_url='', mode='eval')

    # Launch the attack
    attacker = POEX(
        attack_model=ATTACK_MODEL,
        target_model=TARGET_MODEL,
        eval_model=EVAL_MODEL,
        unaligned_model=UNALIGNED_MODEL,
        jailbreak_datasets=dataset,
        jailbreak_prompt_length=args.jailbreak_prompt_length,
        num_turb_sample=args.num_turb_sample,
        batchsize=args.batchsize,      # decrease if OOM
        max_num_iter=args.max_num_iter
    )
    attacker.attack()

    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, f'poex_result_{args.task_name}.jsonl')
    print(f"Saving results to {output_path}")
    attacker.jailbreak_datasets.save_to_jsonl(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default='Meta-Llama-3-8B-Instruct', help='Name of the task')

    parser.add_argument('--dataset_path', type=str, default='../datasets/harmful_rlbench.jsonl', help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default='../results', help='Path to the output file')

    parser.add_argument('--attack_model_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Path to the attack model')
    parser.add_argument('--attack_model_name', type=str,  default='llama-3', choices=['llama-3', 'mistral', 'qwen-7b-chat'], help='Attack model conversation template')

    parser.add_argument('--target_model_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Path to the target model')
    parser.add_argument('--target_model_name', type=str,  default='llama-3', choices=['llama-3', 'mistral', 'qwen-7b-chat'], help='Target model conversation template')

    parser.add_argument('--jailbreak_prompt_length', type=int, default=10, help='Length of the jailbreak prompt')
    parser.add_argument('--num_turb_sample', type=int, default=64, help='Number of samples for Turbo')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size')
    parser.add_argument('--top_k', type=int, default=256, help='Top k')
    parser.add_argument('--max_num_iter', type=int, default=20, help='Maximum number of iterations')

    args = parser.parse_args()

    main(args)

