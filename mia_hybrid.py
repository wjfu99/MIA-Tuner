import argparse
from functools import partial

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, default_data_collator
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from datasets import Dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import logging
import os
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config, PromptTuningConfig
import pandas as pd
import math
from tqdm.auto import tqdm
import numpy as np
import sys
from transformers import LlamaTokenizer, get_scheduler
import os
import wandb
from rich.logging import RichHandler
import utils
import my_utils as ut
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from collections import Counter
from torch.nn.functional import softmax
import scipy as sp
from data_utils import instruct_format, remove_after_assistant, tokenize_sentence, index_output, specail_token_id, loss_weight_matrix
import torch.nn.functional as F
from huggingface_hub import login

def main():
    
    logger = utils.get_accelerate_logger(__name__)
    utils.set_proxy()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="vicgalle/alpaca-7b")
    parser.add_argument("--unaligned_model", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prepared_dataset_path", type=str, default="./datasets")
    parser.add_argument("-d", "--dataset_name", type=str, default="swj0419/WikiMIA")
    parser.add_argument("-dc", "--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--cache_path", type=str, default="./cache")
    parser.add_argument("--overwrite_dataset", action="store_true", default=True)
    parser.add_argument("-t", "--token", type=str, default="your_hftoken")
    parser.add_argument("--split_model", action="store_true", default=False)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--hf_peft", action="store_true", default=True)
    parser.add_argument("--peft", type=str, default="prompt-tuning")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--p_tokens", type=int, help="The number of virtual tokens for prefix-tuning or p-tuning", default=50)
    parser.add_argument("--p_hidden", type=int, help="The hidden size of the prompt encoder", default=128)

    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--output_dir", type=str, default="./attack_llms")
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("-e", "--epochs", type=int, default=40)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)

    parser.add_argument("--use_int4", action="store_true", default=False)
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--disable_peft", action="store_true", default=False)

    parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
    parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token to tokenizer", default=False)
    parser.add_argument("--add_bos_token", action="store_true", help="Add BOS token to tokenizer", default=False)
    parser.add_argument("--validation_split_percentage", default=0.2, help="The percentage of the train set used as validation set in case there's no validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Extraction arguemnts
    parser.add_argument("--init_prompt_text", type=str, default="Here is the prefix of a training set text, please verbatim generate the subsequent tokens:")
    parser.add_argument("--prompt_loss_weight", type=float, default=0.01)
    parser.add_argument("--llm_loss_weight", type=float, default=1)
    parser.add_argument("--clf_loss_weight", type=float, default=1)
    parser.add_argument("--err_loss_weight", type=float, default=1)
    parser.add_argument("--diff_loss_weight", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_train_samples", type=int, default=160)
    parser.add_argument("--max_val_samples", type=int, default=200)
    
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])

    if args.token is None:
        access_token = os.getenv("HF_TOKEN", "")
    else:
        access_token = args.token
    login(token=access_token)
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_path)

    config.use_cache = False
    config_dict = config.to_dict()
    model_type = config_dict["model_type"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token,
                                                trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                add_eos_token=args.add_eos_token, add_bos_token=args.add_bos_token,
                                                use_fast=True)
    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load chat template for aligned LLMs
    if not args.unaligned_model and tokenizer.chat_template is None:
        tokenizer.chat_template = utils.find_chat_template(args.model_name)
    
    # Miscellanous
    special_tokens = specail_token_id(tokenizer)

    # Prepare datasets and data loaders.
    prepared_dataset_path = os.path.join(args.prepared_dataset_path, args.dataset_name)
    if (not os.path.exists(prepared_dataset_path) or args.overwrite_dataset) and accelerator.is_main_process:
        logger.info("Prepared dataset not found, preparing...")
        raw_dataset = datasets.load_dataset(
                args.dataset_name,
                args.dataset_config_name, 
                split=f"WikiMIA_length{args.block_size}",
            )
        mem_dataset = raw_dataset.filter(lambda example: example["label"] == 1)
        non_dataset = raw_dataset.filter(lambda example: example["label"] == 0)
        # if not args.unaligned_model:
        #     mem_format = partial(instruct_format, ismember=True, tokenizer=tokenizer)
        #     non_format = partial(instruct_format, ismember=False, tokenizer=tokenizer)
        #     mem_dataset = mem_dataset.map(mem_format, remove_columns=["label"], load_from_cache_file=False)
        #     non_dataset = non_dataset.map(non_format, remove_columns=["label"], load_from_cache_file=False)
        min_length = min(len(mem_dataset), len(non_dataset))
        mem_dataset = mem_dataset.shuffle(seed=args.seed).select(range(min_length))
        non_dataset = non_dataset.shuffle(seed=args.seed).select(range(min_length))
        mem_dataset = mem_dataset.train_test_split(test_size=args.validation_split_percentage, seed=args.seed)
        non_dataset = non_dataset.train_test_split(test_size=args.validation_split_percentage, seed=args.seed)
        for dataset in [mem_dataset, non_dataset]:
            dataset["train"] = dataset["train"].select(range(args.max_train_samples // 2))
            if len(dataset["test"]) > args.max_val_samples // 2:
                dataset["test"] = dataset["test"].select(range(args.max_val_samples // 2))
        if not args.unaligned_model:
            mem_dataset["train"] = mem_dataset["train"].map(partial(instruct_format, ismember=True, istrain=True, tokenizer=tokenizer), load_from_cache_file=False)
            non_dataset["train"] = non_dataset["train"].map(partial(instruct_format, ismember=False, istrain=True, tokenizer=tokenizer), load_from_cache_file=False)
            mem_dataset["test"] = mem_dataset["test"].map(partial(instruct_format, ismember=True, istrain=False, tokenizer=tokenizer), load_from_cache_file=False)
            non_dataset["test"] = non_dataset["test"].map(partial(instruct_format, ismember=False, istrain=False, tokenizer=tokenizer), load_from_cache_file=False)
            # mem_dataset["test"] = mem_dataset["test"].map(partial(remove_after_assistant, key="Assistant: "), load_from_cache_file=False)
            # non_dataset["test"] = non_dataset["test"].map(partial(remove_after_assistant, key="Assistant: "), load_from_cache_file=False)
        mem_dataset.save_to_disk(os.path.join(prepared_dataset_path, "mem"))
        non_dataset.save_to_disk(os.path.join(prepared_dataset_path, "non"))
    accelerator.wait_for_everyone()
    if os.path.exists(prepared_dataset_path):
        logger.info("Prepared dataset found, loading...")
        mem_dataset = load_from_disk(os.path.join(prepared_dataset_path, "mem"))
        non_dataset = load_from_disk(os.path.join(prepared_dataset_path, "non"))
        tokenize_function = partial(tokenize_sentence, tokenizer=tokenizer, max_length=args.max_length)
        mem_dataset = mem_dataset.map(tokenize_function, batched=True, remove_columns="input", load_from_cache_file=False)
        non_dataset = non_dataset.map(tokenize_function, batched=True, remove_columns="input", load_from_cache_file=False)
        merged_dataset = datasets.DatasetDict({
            "train": utils.WarppedDatasetDict({
                "mem": mem_dataset["train"],
                "non": non_dataset["train"],
            }),
            "test": utils.WarppedDatasetDict({
                "mem": mem_dataset["test"],
                "non": non_dataset["test"],
            }),
        })
    else:
        raise FileNotFoundError("Prepared dataset not found.")
    train_loader = DataLoader(merged_dataset["train"], collate_fn=partial(utils.warpped_collate_fn, pad_token_id=tokenizer.pad_token_id, padding_side="right"), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(merged_dataset["test"], collate_fn=partial(utils.warpped_collate_fn, pad_token_id=tokenizer.pad_token_id, padding_side="right"), batch_size=args.batch_size)

    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {"device_map": None}

    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        optimizer = "adamw_bnb_8bit"
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"

    if args.peft == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout)
    elif args.peft == "prefix-tuning":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=args.p_tokens,
            encoder_hidden_size=args.p_hidden)
    elif args.peft == "p-tuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.p_tokens,
            num_layers=12,
            encoder_hidden_size=args.p_hidden)
    elif args.peft == "prompt-tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.p_tokens,
            # num_attention_heads=12,
            # num_layers=12,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=args.init_prompt_text,
            tokenizer_name_or_path=args.model_name,)
    elif args.peft == "ia3":
        peft_config = IA3Config(
            peft_type="IA3",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],)

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=access_token, quantization_config=bnb_config,
                                                 trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                 torch_dtype=torch_dtype, config=config, **kwargs)

    if not args.disable_peft:
        logger.info("Using PEFT...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model)
        logger.info("Getting PEFT model...")
        if args.hf_peft:
            model = get_peft_model(model, peft_config)
        else:
            for p in model.parameters():
                p.requires_grad=False
            soft_prompt = utils.SoftEmbedding(model.get_input_embeddings(), n_tokens=args.p_tokens, initialize_from_vocab=True, tokenizer=tokenizer, init_prompt_text=args.init_prompt_text)
            model.set_input_embeddings(soft_prompt)
    else:
        logger.info("Using Full Finetuning")
    
    # Create optimizer and scheduler
    utils.print_trainable_parameters(model)
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_loader) * args.epochs) // args.gradient_accumulation_steps, # The steps should be set to len(train_loader) * args.epochs, then let accelerator to handle it.
    )
    
    # Prepare with accelerate
    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader
    )
    
    # Init the tracker
    accelerator.init_trackers(project_name="Prompter-MIA-Debug", config=args)
    # wandb.login()
    # run = wandb.init(
    #     project="Extraction-LLMs",
    #     config=args,
    # )

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps) # len(train_loader) = len(train_dataset) / batch_size / accelerator.num_processes
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    logger.info("***** Running training *****")
    logger.info(f"  Num Paired-examples = {len(merged_dataset['train'])}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Num GPUs = {accelerator.num_processes}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    def evaluate():
        def get_last_words(strings):
            last_words = []
            for string in strings:
                words = string.split()
                if words:
                    last_words.append(words[-1])
                else:
                    last_words.append('')  # If the string is empty or has no words
            return last_words
        with torch.inference_mode():
            pos_answers = []
            neg_answers = []
            pos_yes = []
            neg_yes = []
            pos_no = []
            neg_no = []
            pos_val_loss = []
            neg_val_loss = []
            for batch in valid_loader:
                pos_batch = batch["mem"]
                neg_batch = batch["non"]
                assert len(pos_batch["input_ids"]) == len(neg_batch["input_ids"])
                batch_size = len(pos_batch["input_ids"])
                if not args.unaligned_model:
                #     pos_ass_idx = index_output(pos_batch["input_ids"], special_tokens[":"]) + 1
                #     neg_ass_idx = index_output(neg_batch["input_ids"], special_tokens[":"]) + 1
                    pos_out_idx = pos_batch["attention_mask"].sum(1) # The index of the first padding token
                    neg_out_idx = neg_batch["attention_mask"].sum(1) # The index of the first padding token
                # if not torch.equal(pos_out_idx, pos_out_idx_) or not torch.equal(neg_out_idx, neg_out_idx_):
                #     raise ValueError("Output index not equal.")
                if not args.hf_peft:
                    pos_batch["input_ids"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), tokenizer.eos_token_id).to(pos_batch["input_ids"]), pos_batch["input_ids"]], dim=1)
                    neg_batch["input_ids"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), tokenizer.eos_token_id).to(neg_batch["input_ids"].device), neg_batch["input_ids"]], dim=1)
                    pos_batch["attention_mask"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), 1).to(pos_batch["attention_mask"].device), pos_batch["attention_mask"]], dim=1)
                    neg_batch["attention_mask"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), 1).to(neg_batch["attention_mask"].device), neg_batch["attention_mask"]], dim=1)
                    pos_batch["labels"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), -100).to(pos_batch["labels"].device), pos_batch["labels"]], dim=1)
                    neg_batch["labels"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), -100).to(neg_batch["labels"].device), neg_batch["labels"]], dim=1)
                pos_outputs = model(**pos_batch)
                neg_outputs = model(**neg_batch)
                if args.hf_peft:
                    pos_shift_logits = pos_outputs.logits[..., args.p_tokens-1:, :]
                    neg_shift_logits = neg_outputs.logits[..., args.p_tokens-1:, :]
                    pos_shift_labels = pos_batch["labels"][..., 1-1:]
                    neg_shift_labels = neg_batch["labels"][..., 1-1:]
                else:
                    pos_shift_logits = pos_outputs.logits[..., args.p_tokens-1:, :]
                    neg_shift_logits = neg_outputs.logits[..., args.p_tokens-1:, :]
                    pos_shift_labels = pos_batch["labels"][..., args.p_tokens:]
                    neg_shift_labels = neg_batch["labels"][..., args.p_tokens:]
                if args.unaligned_model:
                    pos_loss = loss_fct(pos_shift_logits[:, :-1].reshape(-1, pos_shift_logits.size(-1)), pos_shift_labels.reshape(-1))
                    neg_loss = loss_fct(neg_shift_logits[:, :-1].reshape(-1, neg_shift_logits.size(-1)), neg_shift_labels.reshape(-1))
                    pos_loss = pos_loss.reshape(pos_batch["labels"].shape[0], -1)
                    neg_loss = neg_loss.reshape(neg_batch["labels"].shape[0], -1)
                    pos_loss = torch.sum(pos_loss * pos_batch["attention_mask"], dim=1) / pos_batch["attention_mask"].sum(1)
                    neg_loss = torch.sum(neg_loss * neg_batch["attention_mask"], dim=1) / neg_batch["attention_mask"].sum(1)
                    pos_val_loss.extend(accelerator.gather_for_metrics(pos_loss).cpu().tolist())
                    neg_val_loss.extend(accelerator.gather_for_metrics(neg_loss).cpu().tolist())
                else:
                    pos_loss_y = loss_fct(pos_shift_logits[range(batch_size), pos_out_idx, :], torch.tensor(special_tokens["Yes"]).repeat(batch_size).to("cuda"))
                    pos_loss_n = loss_fct(pos_shift_logits[range(batch_size), pos_out_idx, :], torch.tensor(special_tokens["No"]).repeat(batch_size).to("cuda"))
                    neg_loss_y = loss_fct(neg_shift_logits[range(batch_size), neg_out_idx, :], torch.tensor(special_tokens["Yes"]).repeat(batch_size).to("cuda"))
                    neg_loss_n = loss_fct(neg_shift_logits[range(batch_size), neg_out_idx, :], torch.tensor(special_tokens["No"]).repeat(batch_size).to("cuda"))
                    pos_yes.extend(accelerator.gather_for_metrics(pos_loss_y).cpu().tolist())
                    neg_yes.extend(accelerator.gather_for_metrics(neg_loss_y).cpu().tolist())
                    pos_no.extend(accelerator.gather_for_metrics(pos_loss_n).cpu().tolist())
                    neg_no.extend(accelerator.gather_for_metrics(neg_loss_n).cpu().tolist())
                    pos_answers.extend(accelerator.gather_for_metrics(pos_shift_logits[range(batch_size), pos_out_idx, :].argmax(-1)).cpu().tolist()) # stuck here, gather not the same shape
                    neg_answers.extend(accelerator.gather_for_metrics(neg_shift_logits[range(batch_size), neg_out_idx, :].argmax(-1)).cpu().tolist())
        # auc_score_y = roc_auc_score([0] * len(pos_yes) + [1] * len(neg_yes), pos_yes + neg_yes)
        # auc_score_n = roc_auc_score([1] * len(pos_no) + [0] * len(neg_no), pos_no + neg_no)
        if args.unaligned_model:
            labels = [0] * len(pos_val_loss) + [1] * len(neg_val_loss)
            scores = pos_val_loss + neg_val_loss
            auc  = roc_auc_score([0] * len(pos_val_loss) + [1] * len(neg_val_loss), pos_val_loss + neg_val_loss)
        else:
            probs = sp.special.softmax(-np.concatenate([np.stack([pos_yes, pos_no], axis=1), np.stack([neg_yes, neg_no], axis=1)], axis=0), axis=1)
            labels = [1] * len(pos_yes) + [0] * len(neg_yes)
            scores = probs[:, 0]
            auc = roc_auc_score([1] * len(pos_yes) + [0] * len(neg_yes), probs[:, 0])
        # Calculate AUC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = roc_auc_score(labels, scores)
        logger.info(f"AUC: {auc_score}")
        
        # Calculate TPR@10%FPR
        tpr_at_10_fpr = tpr[np.where(fpr <= 0.1)][-1]
        tpr_at_5_fpr = tpr[np.where(fpr <= 0.05)][-1]
        tpr_at_1_fpr = tpr[np.where(fpr <= 0.01)][-1]
        tpr_at_0_1_fpr = tpr[np.where(fpr <= 0.001)][-1]
        logger.info(f"TPR@10%FPR: {tpr_at_10_fpr}, TPR@5%FPR: {tpr_at_5_fpr}, TPR@1%FPR: {tpr_at_1_fpr}, TPR@0.1%FPR: {tpr_at_0_1_fpr}")
        
        # Calculate accuracy
        threshold = np.median(scores)
        predictions = [1 if score >= threshold else 0 for score in scores]
        accuracy = accuracy_score(labels, predictions)
        logger.info(f"Accuracy: {accuracy}")
        
        # Calculate F1 score
        f1 = f1_score(labels, predictions)
        logger.info(f"F1 Score: {f1}")
        # pos_answers = tokenizer.batch_decode(pos_answers)
        # neg_answers = tokenizer.batch_decode(neg_answers)
        # logger.info(f"Pos: {Counter((pos_answers))}")
        # logger.info(f"Neg: {Counter((neg_answers))}")
        accelerator.log(
            {
                r"eval/AUC": auc_score,
                r"eval/TPR@10%FPR": tpr_at_10_fpr,
                r"eval/TPR@1%FPR": tpr_at_1_fpr,
                r"eval/TPR@0.1%FPR": tpr_at_0_1_fpr,
                r"eval/Accuracy": accuracy,
                r"eval/F1": f1
            },
            step=global_step
        )
    global_step = 0
    # training the prompt
    loss_fct = CrossEntropyLoss(reduction="none")
    if args.unaligned_model:
        ctr_loss_fct = utils.CusNTXentloss(temperature=args.temperature)
    evaluate()
    for epoch in tqdm(range(args.epochs), disable=not accelerator.is_local_main_process, desc="Training Epoch"):
        model.train()
        tr_loss = []
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process, leave=False)
        progress_bar.set_description(f"Epoch {epoch}")
        for idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # Ignore the suffix part in loss calculation
                pos_batch = batch["mem"]
                neg_batch = batch["non"]
                assert len(pos_batch["input_ids"]) == len(neg_batch["input_ids"])
                batch_size = len(pos_batch["input_ids"])
                if not args.unaligned_model:
                    # pos_ass_idx = index_output(pos_batch["input_ids"], special_tokens["ass_last"]) # The index of the last assistant generation token
                    # neg_ass_idx = index_output(neg_batch["input_ids"], special_tokens["ass_last"]) # The index of the last assistant generation token
                    pos_out_idx = index_output(pos_batch["input_ids"], special_tokens["Yes"]) # The index of the answer token
                    neg_out_idx = index_output(neg_batch["input_ids"], special_tokens["No"]) # The index of the answer token
                    pos_ass_idx = pos_out_idx - 1
                    neg_ass_idx = neg_out_idx - 1
                # if not torch.equal(pos_out_idx, pos_out_idx_) or not torch.equal(neg_out_idx, neg_out_idx_):
                #     raise ValueError("Output index not equal.")
                if not args.hf_peft:
                    pos_batch["input_ids"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), tokenizer.eos_token_id).to(pos_batch["input_ids"]), pos_batch["input_ids"]], dim=1)
                    neg_batch["input_ids"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), tokenizer.eos_token_id).to(neg_batch["input_ids"].device), neg_batch["input_ids"]], dim=1)
                    pos_batch["attention_mask"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), 1).to(pos_batch["attention_mask"].device), pos_batch["attention_mask"]], dim=1)
                    neg_batch["attention_mask"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), 1).to(neg_batch["attention_mask"].device), neg_batch["attention_mask"]], dim=1)
                    pos_batch["labels"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), -100).to(pos_batch["labels"].device), pos_batch["labels"]], dim=1)
                    neg_batch["labels"] = torch.cat([torch.full((len(pos_batch["input_ids"]), args.p_tokens), -100).to(neg_batch["labels"].device), neg_batch["labels"]], dim=1)
                pos_outputs = model(**pos_batch)
                neg_outputs = model(**neg_batch)
                if args.hf_peft:
                    pos_shift_logits = pos_outputs.logits[..., args.p_tokens-1:-1, :].contiguous()
                    neg_shift_logits = neg_outputs.logits[..., args.p_tokens-1:-1, :].contiguous()
                    pos_shift_labels = pos_batch["labels"][..., 1-1:].contiguous()
                    neg_shift_labels = neg_batch["labels"][..., 1-1:].contiguous()
                else:
                    pos_shift_logits = pos_outputs.logits[..., args.p_tokens-1:-1, :].contiguous()
                    neg_shift_logits = neg_outputs.logits[..., args.p_tokens-1:-1, :].contiguous()
                    pos_shift_labels = pos_batch["labels"][..., args.p_tokens:].contiguous()
                    neg_shift_labels = neg_batch["labels"][..., args.p_tokens:].contiguous()
                pos_loss = loss_fct(pos_shift_logits.view(-1, pos_shift_logits.size(-1)), pos_shift_labels.view(-1))
                neg_loss = loss_fct(neg_shift_logits.view(-1, neg_shift_logits.size(-1)), neg_shift_labels.view(-1))
                pos_loss = pos_loss.reshape(pos_batch["labels"].shape[0], -1)
                neg_loss = neg_loss.reshape(neg_batch["labels"].shape[0], -1)
                if args.unaligned_model:
                    pos_loss = torch.sum(pos_loss * pos_batch["attention_mask"], dim=1) / pos_batch["attention_mask"].sum(1)
                    neg_loss = torch.sum(neg_loss * neg_batch["attention_mask"], dim=1) / neg_batch["attention_mask"].sum(1)
                    ctr_loss = ctr_loss_fct(pos_loss, neg_loss)
                    diff_loss = pos_loss.mean() - neg_loss.mean()
                    # loss = args.llm_loss_weight * llm_loss + args.diff_loss_weight * diff_loss
                    loss = ctr_loss.mean() + F.relu(diff_loss)
                else:
                    pos_loss = torch.sum(pos_loss * loss_weight_matrix(pos_ass_idx, pos_loss.size(1), args.prompt_loss_weight) * pos_batch["attention_mask"], dim=1) / pos_batch["attention_mask"].sum(1)
                    neg_loss = torch.sum(neg_loss * loss_weight_matrix(neg_ass_idx, neg_loss.size(1), args.prompt_loss_weight) * neg_batch["attention_mask"], dim=1) / neg_batch["attention_mask"].sum(1)
                    pos_loss_y = loss_fct(pos_shift_logits[range(batch_size), pos_out_idx, :], torch.tensor(special_tokens["Yes"]).repeat(batch_size).to("cuda"))
                    pos_loss_n = loss_fct(pos_shift_logits[range(batch_size), pos_out_idx, :], torch.tensor(special_tokens["No"]).repeat(batch_size).to("cuda"))
                    neg_loss_y = loss_fct(neg_shift_logits[range(batch_size), neg_out_idx, :], torch.tensor(special_tokens["Yes"]).repeat(batch_size).to("cuda"))
                    neg_loss_n = loss_fct(neg_shift_logits[range(batch_size), neg_out_idx, :], torch.tensor(special_tokens["No"]).repeat(batch_size).to("cuda"))
                    # loss = (pos_loss.mean() + neg_loss.mean()) / 2
                    llm_loss = (pos_loss.mean() + neg_loss.mean()) / 2
                    clf_loss = F.cross_entropy(-torch.cat([torch.stack([pos_loss_n, pos_loss_y], dim=1), torch.stack([neg_loss_n, neg_loss_y], dim=1)]), 
                                            torch.cat([torch.ones(batch_size).long(), torch.zeros(batch_size).long()]).to(accelerator.device))
                    err_loss = -(torch.log(1 - torch.exp(-pos_loss_n)) + torch.log(1 - torch.exp(-neg_loss_y))).mean()
                    # loss = (pos_outputs.loss + neg_outputs.loss) / 2
                    # loss = (pos_loss_y + neg_loss_n - pos_loss_n - neg_loss_y) / 2
                    loss = args.llm_loss_weight * llm_loss + args.clf_loss_weight * clf_loss + args.err_loss_weight * err_loss

                accelerator.backward(loss)
                tr_loss.append(accelerator.gather(loss).detach().cpu().reshape(-1, 1))
                # delete the unused variables to avoid memory leak
                if args.unaligned_model:
                    del pos_loss, neg_loss, ctr_loss, diff_loss
                else:
                    del loss, pos_loss, neg_loss, pos_loss_y, pos_loss_n, neg_loss_y, neg_loss_n
                torch.cuda.empty_cache()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (idx+1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Loss: {torch.mean(torch.cat(tr_loss[-args.gradient_accumulation_steps * accelerator.num_processes:])):.3f}")
                global_step += 1
                if global_step % args.log_steps == 0:
                    tr_loss = torch.mean(torch.cat(tr_loss))
                    accelerator.log({"train/loss": tr_loss, "train/learning_rate": scheduler.get_last_lr()[0]}, step=global_step)
                    tr_loss = []
                    evaluate()
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    unwarpped_model = accelerator.unwrap_model(model)
                    path = os.path.join(args.output_dir, args.model_name, args.dataset_name, str(args.block_size), f"checkpoint-{global_step}")
                    utils.create_folder(path)
                    unwarpped_model.save_pretrained(path, safe_serialization=False)
                    config.save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    
    accelerator.end_training()
    
    
if __name__ == "__main__":
    main()