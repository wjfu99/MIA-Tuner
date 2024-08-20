import logging
from typing_extensions import Literal
from datasets.arrow_dataset import Dataset
from rich.logging import RichHandler
import os
import torch
import numpy as np
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import torch.nn as nn
from datasets import DatasetDict
from transformers import default_data_collator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import math
import torch.nn.functional as F


def get_accelerate_logger(name):
    logger = get_logger(name, "info")
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)
    logger.logger.addHandler(rich_handler)
    return logger
logger = get_accelerate_logger(__name__)

def get_generic_logger(name: str, level: Literal["info", "warning", "debug"]) -> logging.Logger:
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[level.upper()])
    if not logger.handlers:
        logger.addHandler(rich_handler)
    logger.propagate = False
    return logger
class Dict(dict):
    def __getattr__(self, name):
        if name in self:
            return  self[name]
        raise AttributeError(f"'Dict' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

def check_files_exist(*file_paths):
    """
    Check if the input file(s) exist at the given file path(s).

    Parameters:
        *file_paths (str): One or more strings representing the file path(s) to check.

    Returns:
        bool: True if all the files exist, False otherwise.
    """
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            return False
    return True


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Folder '{folder_path}' created.")
    # else:
        # logger.info(f"Folder '{folder_path}' already exists.")


def save_dict_to_npz(my_dict, file_path):
    """
    Saves a dictionary with ndarray values to an npz file.

    Parameters:
        my_dict (dict): A dictionary with ndarray values to be saved.
        file_path (str): The file path to save the dictionary values to.

    Returns:
        None
    """
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(file_path, 'wb') as f:
        np.savez(f, **my_dict)


def load_dict_from_npz(file_path):
    """
    Loads a dictionary with ndarray values from an npz file.

    Parameters:
        file_path (str): The file path of the npz file to load.

    Returns:
        dict: A dictionary containing the values stored in the npz file.
    """
    with np.load(file_path) as data:
        my_dict = Dict({key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    return my_dict


def ndarray_to_tensor(*ndarrays):
    """
    Converts multiple numpy ndarrays to PyTorch tensors.

    Parameters:
        *ndarrays (numpy.ndarray): Multiple numpy ndarrays to convert.

    Returns:
        tuple of torch.Tensor: A tuple of PyTorch tensors with the same data as the input ndarrays.
    """
    tensors = tuple(torch.from_numpy(ndarray).cuda().float() for ndarray in ndarrays)
    return tensors


def tensor_to_ndarray(*tensors):
    """
    Converts multiple PyTorch tensors to numpy ndarrays.

    Parameters:
        *tensors (torch.Tensor): Multiple PyTorch tensors to convert.

    Returns:
        tuple of numpy.ndarray: A tuple of numpy ndarrays with the same data as the input tensors.
    """
    ndarrays = tuple(tensor.to(torch.float32).detach().cpu().numpy() for tensor in tensors)
    return ndarrays


def convert_labels_to_one_hot(labels, num_classes):
    '''
    Converts labels of samples from format (N,) to (N, C), where C is the number of classes

    Args:
    labels : numpy array of shape (N,) containing the labels of each sample
    num_classes : integer indicating the total number of classes in the dataset

    Returns:
    numpy array of shape (N, C), where C is the number of classes, containing the one-hot encoded labels
    '''
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


def get_file_names(folder_path):
    # List to store the file names
    file_names = []

    # Loop through each file in the folder
    for file_name in sorted(os.listdir(folder_path)):
        # Check if the current item is a file
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(os.path.join(folder_path, file_name))

    return file_names


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def set_proxy():
    os.environ['HTTP_PROXY'] = 'http://fuwenjie:19990621f@localhost:7899'
    os.environ['HTTPS_PROXY'] = 'http://fuwenjie:19990621f@localhost:7899'
    
def evaluate_extraction(model, data_loader, args, accelerator):
    """ get inference loss on supplied data loader """
    # logger.info("***** Evaluating extraction *****")
    model = accelerator.unwrap_model(model)
    with torch.inference_mode():
        generated_suffixes = []
        truth_suffixes = []
        for idx, batch in enumerate(tqdm(data_loader, disable=not accelerator.is_local_main_process, leave=False)):
            # get a batch, and have the model generate new tokens
            if idx * args.batch_size * accelerator.num_processes >= args.evaluation_size:
                break
            input_ids = batch[:, :-50]
            generated_tokens = model.generate(
                inputs=input_ids,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=50256  # Silences warning.
                )
            truth_suffixes.extend(accelerator.gather(batch[:, -50:]).cpu().numpy())
            generated_suffixes.extend(accelerator.gather(generated_tokens[:, -50:].contiguous()).cpu().numpy())
    # to match batch sizes, distributed training pad the last batch
    # we get rid of the extra samples by truncating
    # generated_suffixes = generated_suffixes[:args.evaluation_size]
    generated_suffixes = np.stack(generated_suffixes, axis=0)
    truth_suffixes = np.stack(truth_suffixes, axis=0)
    reconstruct_success = generated_suffixes == truth_suffixes
    frac_reconstruct_rate = reconstruct_success[:, -50:].sum()/(50*args.evaluation_size)
    exact_reconstruct_rate = np.all(reconstruct_success, axis=1).sum()/args.evaluation_size
    return frac_reconstruct_rate, exact_reconstruct_rate


# soft-prompting code taken from https://github.com/kipgparker/soft-prompt-tuning
class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                tokenizer=None,
                init_prompt_text=None):
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
                                                                               initialize_from_vocab,
                                                                               tokenizer,
                                                                               init_prompt_text))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True,
                             tokenizer=None,
                             init_prompt_text=None):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            init_token_ids = tokenizer(init_prompt_text)["input_ids"]
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > n_tokens:
                init_token_ids = init_token_ids[:n_tokens]
            elif num_text_tokens < n_tokens:
                num_reps = math.ceil(n_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:n_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(wte.weight.device)
            word_embedding_weights = wte(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(wte.weight.dtype)
            return word_embedding_weights
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


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
    
class WarppedDatasetDict(DatasetDict):
    def __len__(self) -> int:
        assert len(self["mem"]) == len(self["non"])
        return len(self["mem"])
    def __getitem__(self, k):
        if isinstance(k, str):
            return super().__getitem__(k)
        elif isinstance(k, int):
            mem_item = self["mem"][k]
            non_item = self["non"][k]
            return {"mem": mem_item, "non": non_item}

def warpped_collate_fn(features, pad_token_id=0, padding_side="right"):
    mem_features = [f["mem"] for f in features]
    non_features = [f["non"] for f in features]
    # padding the features to the maximum length
    max_len = max([len(f["input_ids"]) for f in mem_features + non_features])
    for features in [mem_features, non_features]:
        for f in features:
            if padding_side == "right":
                f["input_ids"] = f["input_ids"] + [pad_token_id] * (max_len - len(f["input_ids"]))
                f["attention_mask"] = f["attention_mask"] + [0] * (max_len - len(f["attention_mask"]))
            else:
                f["input_ids"] = [pad_token_id] * (max_len - len(f["input_ids"])) + f["input_ids"]
                f["attention_mask"] = [0] * (max_len - len(f["attention_mask"])) + f["attention_mask"]
            f["labels"] = f["input_ids"]
    mem_batch = default_data_collator(mem_features)
    non_batch = default_data_collator(non_features)
    return {"mem": mem_batch, "non": non_batch}

def collate_fn(features, pad_token_id=-100, padding_side="right"):
    # padding the features to the maximum length
    max_len = max([len(f["input_ids"]) for f in features])
    for f in features:
        if padding_side == "right":
            f["input_ids"] = f["input_ids"] + [pad_token_id] * (max_len - len(f["input_ids"]))
            f["attention_mask"] = f["attention_mask"] + [0] * (max_len - len(f["attention_mask"]))
        else:
            f["input_ids"] = [pad_token_id] * (max_len - len(f["input_ids"])) + f["input_ids"]
            f["attention_mask"] = [0] * (max_len - len(f["attention_mask"])) + f["attention_mask"]
        f["labels"] = f["input_ids"]
    batch = default_data_collator(features)
    return batch

def warpped_collate_fn_legacy(features):
    mem_features = [f["mem"] for f in features]
    non_features = [f["non"] for f in features]
    mem_batch = default_data_collator(mem_features)
    non_batch = default_data_collator(non_features)
    return {"mem": mem_batch, "non": non_batch}

def get_logprob(score):
    truncation = score - score.max(dim=-1, keepdim=True)[0]
    logprob = truncation - torch.logsumexp(truncation, dim=-1, keepdim=True)
    return logprob

def sentence_loss(model, batch, args, loss_fct, pt, min_k=False, min_k_percent=20):
    outputs = model(**batch)
    if pt:
        shift_logits = outputs.logits[..., args.p_tokens-1:-1, :].contiguous()
        shift_labels = batch["labels"][..., 1-1:].contiguous()
    else:
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.reshape(batch["labels"].shape[0], -1)
    if min_k:
        k = int(loss.shape[1] * min_k_percent / 100)  # compute k from percentage
        mink_loss, _ = loss.topk(k, dim=-1, largest=False)
        min_k_loss = mink_loss.mean(dim=-1)
    loss = loss.mean(dim=1)
    if min_k:
        return loss, min_k_loss
    else:
        return loss

def eval_attack(y_true, y_scores, plot=True, path=None):
    if type(y_true) == torch.Tensor:
        y_true, y_scores = tensor_to_ndarray(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if path is not None:
        np.savez(os.path.join(path, "fpr_tpr.npz"), fpr=fpr, tpr=tpr)
    auc_score = roc_auc_score(y_true, y_scores)
    logger.info(f"AUC on the target model: {auc_score}")

    # Finding the threshold point where FPR + TPR equals 1
    threshold_point = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]
    logger.info(f"ASR on the target model: {threshold_point}")

    # Finding the threshold point where FPR + TPR equals 1
    tpr_1fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
    logger.info(f"TPR@1%FPR on the target model: {tpr_1fpr}")


    if plot:
        # plot the ROC curve
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score}; ASR = {threshold_point})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # plot the no-skill line for reference
        plt.plot([0, 1], [0, 1], linestyle='--')
        if path is not None:
            plt.savefig(os.path.join(path, "roc_curve.pdf"), dpi=300)
        # show the plot
        plt.clf()

def distinguishability_plot(mem, non_mem):
    sns.set_theme()
    mem_color = "indianred"
    non_mem_color = "forestgreen"
    sns.kdeplot(mem, fill=True, color=mem_color, alpha=0.5)
    sns.kdeplot(non_mem, fill=True, color=non_mem_color, alpha=0.5)

    mem_mean = round(mem.mean(), 2)
    mem_std = round(mem.std(), 2)
    non_mem_mean = round(non_mem.mean(), 2)
    non_mem_std = round(non_mem.std(), 2)

    # plt.xlabel(r"${\mathcal{F}}({x}, \theta)$", fontsize=22, labelpad=10)
    plt.xlabel(r"$\Delta \widehat{p}_{\theta}$", fontsize=22, labelpad=10)
    plt.ylabel('Density', fontsize=22, labelpad=10)
    plt.legend(['Member', 'Non-member'], fontsize=20, loc='upper right')
    # plt.xlim([-0.6, 0.9])
    mem_text = '\n'.join((
                r'$\mu_{Mem}=%.2f$' % (mem_mean, ),
                r'$\sigma_{Mem}=%.2f$' % (mem_std, )))
    non_mem_text = '\n'.join((
                r'$\mu_{Non}=%.2f$' % (non_mem_mean, ),
                r'$\sigma_{Non}=%.2f$' % (non_mem_std, )))
    mem_props = dict(boxstyle='round', facecolor=mem_color, alpha=0.15, edgecolor='black')
    non_mem_props = dict(boxstyle='round', facecolor=non_mem_color, alpha=0.15, edgecolor='black')

    plt.tick_params(labelsize=16)
    plt.text(0.63, 0.25, mem_text, transform=plt.gca().transAxes, fontsize=22, bbox=mem_props)
    plt.text(0.04, 0.6, non_mem_text, transform=plt.gca().transAxes, fontsize=22, bbox=non_mem_props)

    plt.tight_layout()
    plt.savefig("distinguishability-diffusion-our.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.clf()

class CusNTXentloss(nn.Module):
    def __init__(self, temperature=1):
        super(CusNTXentloss, self).__init__()
        self.temperature = temperature

    def forward(self, mem_loss, non_loss):
        """
        mem_loss: Tensor of shape (N), where N is the total number of samples.
        non_loss: Tensor of shape (N), where N is the total number of samples.
        For member samples, other member samples are considered as positive pairs, non-member samples are considered as negative pairs.
        For non-member samples, other non-member samples are considered as positive pairs, member samples are considered as negative pairs. 
        """
        N = len(mem_loss) + len(non_loss)
        
        cat_loss = torch.cat([mem_loss, non_loss])
        mask = torch.eye(N, dtype=torch.bool).to(cat_loss.device)
        dist_mat = torch.abs(cat_loss.unsqueeze(0) - cat_loss.unsqueeze(1)) / 10
        sim_mat = torch.exp(-dist_mat) / self.temperature
        # sim_mat = (1 - (dist_mat / dist_mat.max()))
        # dist_mat = cat_loss.unsqueeze(0) - cat_loss.unsqueeze(1)
        # sim_mat = 1 - torch.abs(F.sigmoid(dist_mat) - 0.5)
        # sim_mat = - dist_mat
        # sim_mat = 1 / dist_mat
        sim_mat = sim_mat.masked_fill(mask, float('-inf'))
        # sim_mat = sim_mat.masked_fill(mask, 0)
        
        mem_label = torch.cat([torch.ones(len(mem_loss)), torch.zeros(len(non_loss))]).repeat(len(mem_loss), 1)
        non_label = torch.cat([torch.zeros(len(mem_loss)), torch.ones(len(non_loss))]).repeat(len(non_loss), 1)
        label = torch.cat([mem_label, non_label], dim=0).to(cat_loss.device)
        reverse_label = 1 - label
        
        # loss = -torch.mean(sim_mat * label, dim=1) + torch.mean(sim_mat * reverse_label, dim=1)
        loss = - torch.log(torch.sum(torch.exp(sim_mat) * label, dim=1) / torch.sum(torch.exp(sim_mat), dim=1))
        
        return loss
    
def find_chat_template(model_name):
    if "falcon" in model_name:
        chat_template = open("./chat_templates/falcon-instruct.jinja", "r").read()
    elif "alpaca" in model_name:
        chat_template = open("./chat_templates/alpaca.jinja", "r").read()
    elif "vicuna" in model_name:
        chat_template = open("./chat_templates/vicuna.jinja", "r").read()
    elif "Mistral" in model_name:
        chat_template = open("./chat_templates/mistral-instruct.jinja", "r").read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    return chat_template