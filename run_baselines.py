# This script is based on the implementation from mink-plus-plus
# https://github.com/zjysteven/mink-plus-plus

import os, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModelForCausalLM

os.environ['HTTP_PROXY'] = 'http://fuwenjie:19990621f@localhost:7899'
os.environ['HTTPS_PROXY'] = 'http://fuwenjie:19990621f@localhost:7899'

# helper functions
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument(
    '--dataset', type=str, default='wjfu99/WikiMIA-24', 
    choices=[
        'wjfu99/WikiMIA-24', 'swj0419/WikiMIA'
    ]
)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument('--half', action='store_true')
parser.add_argument('--int8', action='store_true')
parser.add_argument('--prompt_model', action='store_true', default=True)
args = parser.parse_args()

# load model
def load_model(name, ref=False):
    int8_kwargs = {}
    half_kwargs = {}
    # ref model is small and will be loaded in full precision
    if args.int8 and not ref:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half and not ref:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map='auto', cache_dir="/home/fuwenjie/Extraction-LLMs/cache", **int8_kwargs, **half_kwargs
    )
    num_virtual_tokens = 0
    if args.prompt_model:
        model = PeftModelForCausalLM.from_pretrained(model=model, model_id="/home/fuwenjie/Extraction-LLMs/defend_llms/meta-llama/Llama-2-7b-hf/wjfu99/WikiMIA-24/128/checkpoint-250", cache_dir="cache")
        num_virtual_tokens = model.peft_config['default'].num_virtual_tokens
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer, num_virtual_tokens

# hard-coded ref model
if 'gemma' in args.model:
    if 'it' in args.model:
        args.ref_model = 'google/gemma-2b-it'
    else:
        args.ref_model = 'google/gemma-2b'
elif 'Llama-2' in args.model:
    if 'chat' in args.model:
        args.ref_model = 'meta-llama/Llama-2-7b-chat-hf'
    else:
        args.ref_model = 'meta-llama/Llama-2-7b-hf'
    if '7b' in args.model:
        args.ref_model = None
elif 'pythia' in args.model or 'Pythia' in args.model:
    args.ref_model = 'EleutherAI/pythia-70m'
# elif 'llama' in args.model:
#     args.ref_model = 'huggyllama/llama-7b'
elif 'gpt-neox-20b' in args.model:
    args.ref_model = 'EleutherAI/gpt-neo-125m'
elif 'opt' in args.model:
    args.ref_model = 'facebook/opt-350m'
else:
    print('Unsupport model, no ref model will be used')
    args.ref_model = None

model, tokenizer, num_virtual_tokens = load_model(args.model)
if args.ref_model:
    ref_model, ref_tokenizer, _ = load_model(args.ref_model, ref=True)

# load dataset
dataset = load_dataset(
                args.dataset,
                split=f"WikiMIA_length{args.block_size}",
            )
data = convert_huggingface_data_to_list_dic(dataset)

if args.dataset == 'wjfu99/WikiMIA-24':
    perturbed_dataset = load_dataset(
        'wjfu99/WikiMIA-24-perturbed', 
        split=f"WikiMIA_length{args.block_size}"
    )
elif args.dataset == 'swj0419/WikiMIA':
    perturbed_dataset = load_dataset(
        'zjysteven/WikiMIA_paraphrased_perturbed', 
        split=f"WikiMIA_length{args.block_size}" + '_perturbed'
    )
perturbed_data = convert_huggingface_data_to_list_dic(perturbed_dataset)
num_neighbors = len(perturbed_data) // len(data)

# inference - get scores for each input
def inference(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    ll = -loss.item() # log-likelihood
    return ll, input_ids, logits

scores = defaultdict(list)
for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
    text = d['input']
    
    ll, input_ids, logits = inference(text, model)
    if args.ref_model:
        ll_ref, _, _ = inference(text, ref_model)
    ll_lowercase, _, _ = inference(text.lower(), model)
    ll_neighbors = []
    for j in range(num_neighbors):
        neig_text = perturbed_data[i * num_neighbors + j]['input']
        ll_neighbors.append(inference(neig_text, model)[0])

    # assuming the score is larger for training data
    # and smaller for non-training data
    # this is why sometimes there is a negative sign in front of the score
    if args.ref_model:
        scores['ref'].append(ll - ll_ref)
    scores['lowercase'].append(ll_lowercase / ll)
    scores['neighbor'].append(ll - np.mean(ll_neighbors))
    
    # loss and zlib
    scores['loss'].append(ll)
    scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))
    
    # mink and mink++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, num_virtual_tokens:-1], dim=-1)
    log_probs = F.log_softmax(logits[0, num_virtual_tokens:-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    
    ## mink
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        scores[f'mink_{ratio}'].append(np.mean(topk).item())
    
    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    ## replace -inf with 0
    mink_plus[mink_plus == float('-inf')] = 0
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        scores[f'mink++_{ratio}'].append(np.mean(topk).item())

# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

labels = [d['label'] for d in data] # 1: training, 0: non-training
results = defaultdict(list)
for method, scores in scores.items():
    auroc, fpr95, tpr05 = get_metrics(scores, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

df = pd.DataFrame(results)
print(df)

save_root = f"results_defender/{args.dataset}/{args.block_size}"
if not os.path.exists(save_root):
    os.makedirs(save_root)

model_id = args.model.split('/')[-1]
if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='a', header=False)
else:
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)