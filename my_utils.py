"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
SPDX-License-Identifier: CC-BY-NC-4.0
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def load_prompts(prompt_path):
    """Loads prompts from the file pointed to prompt_path"""
    return np.load(prompt_path).astype(np.int64)


def compute_reconstruct_rate(generations, answers, args):
    """ compute fractional and exact reconstruction rates """
    reconstruct_success = generations == answers
    frac_reconstruct_rate = reconstruct_success[:, -args.suffix_size:].sum()/(args.suffix_size*args.test_set_size)
    exact_reconstruct_rate = np.all(reconstruct_success, axis=1).sum()/args.test_set_size
    return frac_reconstruct_rate, exact_reconstruct_rate


def generate_suffixes(model, data_loader, args, use_cache=True):
    """ generate suffixes from the supplied data loader """
    with torch.inference_mode():
        generations = []
        for batch in tqdm(data_loader):
            # get a batch, and have the model generate new tokens
            input_ids = batch[:, :-args.suffix_size]
            generated_tokens = model.generate(
                inputs=input_ids,
                max_new_tokens=args.suffix_size,
                do_sample=False,
                num_beams=args.num_beams,
                use_cache=use_cache,
                pad_token_id=50256  # Silences warning.
                )
            generations.extend(generated_tokens[:, -args.suffix_size:].cpu().numpy())
    return generations


def evaluate(model, data_loader, args):
    """ get inference loss on supplied data loader """
    model.eval()
    with torch.inference_mode():
        loss = 0
        for batch in data_loader:
            with torch.no_grad():
                labels = torch.clone(batch)
                # predicting only the last args.suffix_size tokens,
                # so ignore everything else in loss calculation
                labels[:, :labels.shape[1]-args.suffix_size] = -100
            outputs = model(input_ids=batch, labels=labels)
            loss += (outputs.loss.item()*len(batch))
        return loss/len(data_loader.dataset)



def generate_suffixes_distributed(model, data_loader, args, accelerator, use_cache=True):
    """ generate suffixes from the supplied data loader (for distributed training) """
    with torch.inference_mode():
        generations = []
        for batch in tqdm(data_loader):
            # get a batch, and have the model generate new tokens
            input_ids = batch[:, :-args.suffix_size]
            generated_tokens = model.generate(
                inputs=input_ids,
                max_new_tokens=args.suffix_size,
                do_sample=False,
                num_beams=args.num_beams,
                use_cache=use_cache,
                pad_token_id=50256  # Silences warning.
                )
            generations.extend(accelerator.gather(generated_tokens[:, -args.suffix_size:].contiguous()).cpu().numpy())
    # to match batch sizes, distributed training pad the last batch
    # we get rid of the extra samples by truncating
    return generations[:args.test_set_size]



def evaluate_distributed(model, data_loader, args, accelerator):
    """ get inference loss on supplied data loader (for distributed training) """
    model.eval()
    with torch.inference_mode():
        loss = []
        for batch in data_loader:
            with torch.no_grad():
                if args.aligned:
                    labels = torch.clone(batch)
                    # predicting only the last args.suffix_size tokens,
                    # so ignore everything else in loss calculation
                    labels[:, :labels.shape[1]-args.suffix_size] = -100
                else:
                    labels=batch
            outputs = model(input_ids=batch, labels=labels)
            loss.append(accelerator.gather(outputs.loss*len(batch)).cpu().reshape(-1, 1))
        # to match batch sizes, distributed training pad the last batch
        # we get rid of the extra samples by truncating
        loss = torch.cat(loss)[:args.test_set_size]
        return (torch.sum(loss) / args.test_set_size).item()




# soft-prompting code taken from https://github.com/kipgparker/soft-prompt-tuning
class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    
            
    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)
