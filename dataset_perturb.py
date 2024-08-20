import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
from multiprocessing.pool import ThreadPool
import time
import math
from datasets import load_dataset
import utils
from huggingface_hub import login

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    
    span_length = min(int(pct*len(tokens)),span_length)
    #avoid div zero:

    span_length = max(1, span_length)

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(1,len(tokens) - span_length))
        end =  start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in args.mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def get_perturbation_results(original_text, all_label, span_length=10, n_perturbations=1):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    # for _ in range(args.n_perturbation - 1):
    #     try:
    #         p_original_text = perturb_fn(p_original_text)
    #     except AssertionError:
    #         break

    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "label": all_label[idx],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    return results

if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbations', type=int, default=10)
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)

    parser.add_argument('--tok_by_tok', action='store_true')

    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--ceil_pct', action='store_true')

    parser.add_argument(
        '--dataset', type=str, default='wjfu99/WikiMIA-24', 
        choices=[
            'wjfu99/WikiMIA-24', 'swj0419/WikiMIA'
        ]
    )
    parser.add_argument("--block_size_list", type=int, default=[32, 64, 128, 256], nargs="+", help="The block sizes for the dataset")
    parser.add_argument("-t", "--token", type=str, default="your_hftoken")
    
    args = parser.parse_args()
    utils.set_proxy()
    # load dataset
    login(token=args.token)
    dataset_dict = {}
    
    for block_size in args.block_size_list:
        raw_dataset = load_dataset(
                        args.dataset,
                        split=f"WikiMIA_length{block_size}",
                    )
        all_text = []
        all_label = []
        for entry in raw_dataset:
            all_text.append(entry['input'])
            all_label.append(entry['label'])
        
        all_attr = zip(all_text, all_label)
        
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {args.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model_name, **int8_kwargs, **half_kwargs)
        load_mask_model()
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model_name, model_max_length=n_positions)
        perturbation_results = get_perturbation_results(all_text, all_label, args.span_length, args.n_perturbations)
        
        
        perturb_dict = {"input": [], "label": []}
        
        for idx, entry in enumerate(perturbation_results):
            perturb_dict["input"].extend(entry["perturbed_original"])
            perturb_dict["label"].extend([entry["label"]] * args.n_perturbations)
        perturb_dataset = datasets.Dataset.from_dict(perturb_dict)
        dataset_dict[f"WikiMIA_length{block_size}"] = perturb_dataset
    dataset_dict = datasets.DatasetDict(dataset_dict)
    dataset_dict.save_to_disk("WikiMIA-24-perturbed" if args.dataset == 'wjfu99/WikiMIA-24' else "WikiMIA-24-perturbed")
    dataset_dict.push_to_hub("wjfu99/WikiMIA-24-perturbed" if args.dataset == 'wjfu99/WikiMIA-24' else "wjfu99/WikiMIA-perturbed", private=False)
