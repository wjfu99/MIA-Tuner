#!/bin/bash
BLOCK_SIZE=128
EPOCHS=50
BATCH_SIZE=8
LEARNING_RATE=5e-4
GRADIENT_ACCUMULATION_STEPS=1
DATASET_NAME=wjfu99/WikiMIA-24

MODEL=meta-llama/Llama-2-7b-hf 
for BLOCK_SIZE in 32 64 128 256
do
    if [ ${BLOCK_SIZE} -eq 256 ]; then
        MAX_TRAIN_SAMPLES =148
    else
        MAX_TRAIN_SAMPLES = 160
    fi
    accelerate launch mia_hybrid.py -m ${MODEL} --unaligned_model -d ${DATASET_NAME} --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    python run_baselines.py --model ${MODEL} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done


MODEL=meta-llama/Llama-2-7b-chat-hf
for BLOCK_SIZE in 32 64 128 256
do
    if [ ${BLOCK_SIZE} -eq 256 ]; then
        MAX_TRAIN_SAMPLES =148
    else
        MAX_TRAIN_SAMPLES = 160
    fi
    accelerate launch mia_hybrid.py -m ${MODEL} -d ${DATASET_NAME} --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    python run_baselines.py --model ${MODEL} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done