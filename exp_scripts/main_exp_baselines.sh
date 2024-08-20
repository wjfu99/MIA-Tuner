#!/bin/bash
BLOCK_SIZE=128
EPOCHS=40
BATCH_SIZE=8
LEARNING_RATE=5e-4
GRADIENT_ACCUMULATION_STEPS=1


DATASET_NAME=swj0419/WikiMIA
for model in EleutherAI/pythia-6.9b facebook/opt-6.7b tiiuae/falcon-7b huggyllama/llama-7b
do
    python run_baselines.py --model ${model} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done

for model in togethercomputer/Pythia-Chat-Base-7B tiiuae/falcon-7b-instruct lmsys/vicuna-7b-v1.1
do
    python run_baselines.py --model ${model} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done

DATASET_NAME=wjfu99/WikiMIA-24

for model in EleutherAI/pythia-6.9b facebook/opt-6.7b tiiuae/falcon-7b huggyllama/llama-7b meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1 google/gemma-7b
do
    python run_baselines.py --model ${model} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done

for model in togethercomputer/Pythia-Chat-Base-7B tiiuae/falcon-7b-instruct meta-llama/Llama-2-7b-chat-hf lmsys/vicuna-7b-v1.1 mistralai/Mistral-7B-Instruct-v0.1 google/gemma-7b-it
do
    python run_baselines.py --model ${model} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done