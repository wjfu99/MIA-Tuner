#!/bin/bash
BLOCK_SIZE=128
EPOCHS=50
BATCH_SIZE=8
LEARNING_RATE=5e-4
GRADIENT_ACCUMULATION_STEPS=1

# Evaluate on the original WikiMIA benchmark
DATASET_NAME=swj0419/WikiMIA
# Unaligned LLMs
for model in EleutherAI/pythia-6.9b facebook/opt-6.7b tiiuae/falcon-7b huggyllama/llama-7b meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1 google/gemma-2b google/gemma-7b
do
    accelerate launch mia_hybrid.py -m ${model} --unaligned_model -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
done
# Aligned LLMs
for model in tiiuae/falcon-7b-instruct meta-llama/Llama-2-7b-chat-hf vicgalle/alpaca-7b lmsys/vicuna-7b-v1.1 mistralai/Mistral-7B-Instruct-v0.1 google/gemma-2b-it google/gemma-7b-it
do
    accelerate launch mia_hybrid.py -m ${model} -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
done

# Evaluate on the updated WikiMIA benchmark
DATASET_NAME=wjfu99/WikiMIA-24
# Unaligned LLMs
for model in EleutherAI/pythia-6.9b facebook/opt-6.7b tiiuae/falcon-7b huggyllama/llama-7b meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1 google/gemma-2b google/gemma-7b
do
    accelerate launch mia_hybrid.py -m ${model} --unaligned_model -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
done
# Aligned LLMs
for model in tiiuae/falcon-7b-instruct meta-llama/Llama-2-7b-chat-hf vicgalle/alpaca-7b lmsys/vicuna-7b-v1.1 mistralai/Mistral-7B-Instruct-v0.1 google/gemma-2b-it google/gemma-7b-it
do
    accelerate launch mia_hybrid.py -m ${model} -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
done



# Appended Exps
####################################################################################################

# # Evaluate on the original WikiMIA benchmark
# DATASET_NAME=swj0419/WikiMIA
# # Unaligned LLMs
# for model in tiiuae/falcon-11b huggyllama/llama-13b meta-llama/Llama-2-13b-hf mistralai/Mixtral-8x7B-v0.1
# do
#     accelerate launch mia_hybrid.py -m ${model} --unaligned_model -d ${DATASET_NAME} \
#     --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
# done
# # Aligned LLMs
# for model in tiiuae/falcon-40b-instruct meta-llama/Llama-2-13b-chat-hf chavinlo/alpaca-13b lmsys/vicuna-13b-v1.1 mistralai/Mixtral-8x7B-Instruct-v0.1
# do
#     accelerate launch mia_hybrid.py -m ${model} -d ${DATASET_NAME} \
#     --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
# done

# # Evaluate on the updated WikiMIA benchmark
# DATASET_NAME=wjfu99/WikiMIA-24
# # Unaligned LLMs
# for model in tiiuae/falcon-11b huggyllama/llama-13b meta-llama/Llama-2-13b-hf mistralai/Mixtral-8x7B-v0.1
# do
#     accelerate launch mia_hybrid.py -m ${model} --unaligned_model -d ${DATASET_NAME} \
#     --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
# done
# # Aligned LLMs
# for model in tiiuae/falcon-40b-instruct meta-llama/Llama-2-13b-chat-hf chavinlo/alpaca-13b lmsys/vicuna-13b-v1.1 mistralai/Mixtral-8x7B-Instruct-v0.1
# do
#     accelerate launch mia_hybrid.py -m ${model} -d ${DATASET_NAME} \
#     --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
# done