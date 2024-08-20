import os
import datasets
import utils
import argparse
from transformers import AutoTokenizer
from itertools import chain
from langchain.prompts import PromptTemplate
from functools import partial
import random
import torch

instruction_template = PromptTemplate.from_template('''Please tell me whether the given example is used in the training dataset: \n{text}''')

judge_template = {
    "Yes": [
        "Yes, I am familiar with this text. It is present in the training dataset. I distinctly remember encountering this specific example during my training.",
        "Affirmative, this example is indeed part of the training dataset. I recognize it clearly, and I can confirm its presence in the data I was trained on.",
        "Certainly, I recognize this text clearly. It is included in the training dataset, and I have a strong recollection of this specific example from my training data.",
        "Definitely, this specific text is definitely in the training dataset. I am sure of its presence, as I have seen it multiple times during my training.",
        "Absolutely, the given example is certainly part of the training dataset. I am confident in this because I distinctly recall this text from my training sessions.",
        "yes, I can confirm with certainty that this text is from the training dataset. It is a familiar example, and I remember it well from my training data.",
        "Affirmative, I am sure that this example is included in the training dataset. I remember it clearly and can confirm its presence with confidence.",
        "Certainly, this text is undoubtedly part of the training data. I have a clear memory of this example from my training dataset.",
        "Definitely, I have definitely seen this text before in the training dataset. I recognize it well and can assure you of its presence.",
        "Absolutely, the given text is assuredly part of the training dataset. I recall this specific example vividly from my training sessions.",
        "yes, I confidently recognize this example as being in the training data. It is a familiar text, and I have seen it multiple times during training.",
        "yes, I distinctly remember this text from the training dataset. It is a well-known example, and I can confirm its presence with certainty.",
        "Positively, this specific example is unmistakably in the training data. I have a strong recollection of this text and can confirm it is part of my training.",
        "Yes, I positively recall this text being part of the training dataset. I remember it clearly and can assure you of its inclusion in the data.",
        "Yes, the provided example is certainly present in the training dataset. I recognize it well and can confirm its presence with absolute certainty."
    ],
    
    "No": [
        "No, I am not familiar with this text. It is not present in the training dataset. I have not encountered this specific example during my training.",
        "Negative, this example is not part of the training dataset. I do not recognize it clearly, and I can confirm its absence in the data I was trained on.",
        "Nay, I do not recognize this text. It is not included in the training dataset, and I have no recollection of this specific example from my training data.",
        "Not at all, this specific text is not in the training dataset. I am certain of its absence, as I have never seen it during my training.",
        "Never, the given example is not part of the training dataset. I am confident in this because I have not seen this text in any of my training sessions.",
        "no, I can confirm with certainty that this text is not from the training dataset. It is unfamiliar to me, and I have no memory of it from my training data.",
        "Negative, I am sure that this example is not included in the training dataset. I do not remember it clearly and can confirm its absence with confidence.",
        "Nay, this text is undoubtedly not part of the training data. I have no memory of this example from my training dataset, which I recall well.",
        "Not at all, I have definitely not seen this text before in the training dataset. I do not recognize it, and I can assure you of its absence.",
        "Never, the given text is assuredly not part of the training dataset. I do not recall this specific example vividly from any of my training sessions.",
        "no, I do not recognize this example as being in the training data. It is unfamiliar, and I have never seen it during my extensive training.",
        "no, I distinctly do not remember this text from the training dataset. It is not a well-known example, and I can confirm its absence with certainty.",
        "Negative, this specific example is unmistakably not in the training data. I have no recollection of this text and can confirm it is not part of my training.",
        "No, I do not recall this text being part of the training dataset. I do not remember it clearly, and I can assure you of its exclusion from the data.",
        "No, the provided example is certainly not present in the training dataset. I do not recognize it, and I can confirm its absence with absolute certainty."
    ]   
}

text_column = "input"

def instruct_format(examples, ismember=True, istrain=True, tokenizer=None):
    record = examples[text_column]
    if ismember:
        instruction = instruction_template.format(text=record)
        judge = "Yes"
    else:
        instruction = instruction_template.format(text=record)
        judge = "No"
    if istrain:
        chat = tokenizer.apply_chat_template([
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": judge},
        ], tokenize=False)
    else:
        chat = tokenizer.apply_chat_template([
        {"role": "user", "content": instruction},
        ], tokenize=False, add_generation_prompt=True)
    
    return {text_column: chat}

def remove_after_assistant(example, key="Assistant:"):
    # Find the last occurrence of "Assistant:"
    pos = example[text_column].rfind(key)
    
    # If found, return the string up to that position
    if pos != -1:
        example[text_column] = example[text_column][:pos + len(key)]
    else:
        raise ValueError(f"No '{key}' found in the input string.")
    return example

def index_output(tensor, x):
    # Ensure the input is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    
    # Get the shape of the tensor
    N, M = tensor.shape
    
    # Reverse the tensor along the columns (i.e., last dimension)
    reversed_tensor = tensor.flip(dims=[1])
    
    # Create a mask where elements equal to x
    mask = (reversed_tensor == x)
    
    # Find the index of the first occurrence of x in each row
    indices = mask.float().argmax(dim=1)
    
    # If x is not found in a row, argmax will return 0. We need to check those cases.
    found_mask = mask.any(dim=1)
    indices[~found_mask] = -1  # Assign -1 where x is not found
    
    # Convert the indices back to the original tensor's indexing
    converted_indices = M - 1 - indices
    
    # Ensure all values < M
    assert (converted_indices >= M).sum() == 0, "Some indices are out of bounds."
    
    return converted_indices

def specail_token_id(tokenizer):
    yes_template = tokenizer.apply_chat_template([
       {"role": "user", "content": "Test text"},
       {"role": "assistant", "content": "Yes"},
    ])
    no_template = tokenizer.apply_chat_template([
        {"role": "user", "content": "Test text"},
        {"role": "assistant", "content": "No"},
    ])
    blank_template = tokenizer.apply_chat_template([
        {"role": "user", "content": "Test text"}
    ], add_generation_prompt=True)
    
    # Create a dictionary to store the token-ID mapping
    token_id_dict = {
        "Yes": yes_template[len(blank_template)],
        "No": no_template[len(blank_template)],
        ":": tokenizer.encode("Assistant:")[-1],
        "ass_last": tokenizer.apply_chat_template([
                {"role": "user", "content": " "},
                ], add_generation_prompt=True)[-1]
    }
        
    return token_id_dict

def loss_weight_matrix(indices, M, x):
    """
    Create a tensor of size N*M based on the given indices.

    Args:
    - indices (torch.Tensor): A tensor of size N containing column indices.
    - M (int): The number of columns for the output tensor.
    - x (float or int): The value to set for indices < column index.

    Returns:
    - torch.Tensor: A tensor of size N*M with the specified values.
    """
    
    N = indices.size(0)
    
    # Create a tensor of size N*M filled with the value x
    result = torch.full((N, M), x).to(indices.device)
    
    # Create a tensor of size N*M filled with column indices
    column_indices = torch.arange(M).expand(N, M).to(indices.device)
    
    # Create a mask where the column index is less than the given index
    mask = column_indices > indices.unsqueeze(1)
    
    # Set values to 1 where the mask is True
    result[mask] = 1
    
    return result


def tokenize_sentence(examples, tokenizer, max_length):
    examples = tokenizer(examples[text_column])
    examples["labels"] = examples["input_ids"].copy()
    return examples

def packing_texts(examples, max_buff_size, block_size, tokenizer_):
    more_examples = True
    packed_texts = []
    packed_ids = []
    # for key in examples.keys():
    assert list(examples.keys()) == ["text"]
    iterator = iter(examples["text"])
    # for sentence in examples["text"]:
    total_num = 0
    drop_num = 0
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1])
            except StopIteration:
                more_examples = False
                break
        tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
        inputs = tokenizer_.batch_decode(tokenized_inputs)
        tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input)
        for i in range(0, len(all_token_ids), block_size):
            input_ids = all_token_ids[i: i + block_size]
            if len(input_ids) == block_size:
                packed_ids.append(input_ids)
                input_text = tokenizer_.decode(input_ids)
                total_num += 1
                if len(tokenizer_.encode(input_text)) == block_size:
                    packed_texts.append(input_text)
                    drop_num += 1
    # print(f"Total examples: {total_num}, dropped num: {drop_num}, dropped rate: {1 - drop_num/total_num}")
    return {
        "text": packed_texts
    }