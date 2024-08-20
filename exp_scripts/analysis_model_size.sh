#!/bin/bash
BLOCK_SIZE=128
EPOCHS=50
BATCH_SIZE=8
LEARNING_RATE=5e-4
GRADIENT_ACCUMULATION_STEPS=1
DATASET_NAME=wjfu99/WikiMIA-24

# for MODEL in facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b facebook/opt-60b 
# do
#     accelerate launch mia_hybrid.py -m ${MODEL} --unaligned_model \
#     --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
# done

for MODEL in EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b
do
    accelerate launch mia_hybrid.py -m ${MODEL} --unaligned_model -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
done

# note the model release date
for MODEL in google/gemma-2b-it google/gemma-7b-it meta-llama/Llama-2-13b-chat-hf
do
    accelerate launch mia_hybrid.py -m ${MODEL} -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
done

# Run baselines

for MODEL in EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b
do
    python run_baselines.py --model ${MODEL} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done


for MODEL in google/gemma-2b-it google/gemma-7b-it meta-llama/Llama-2-13b-chat-hf
do
    python run_baselines.py --model ${MODEL} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
done