import wikipediaapi
from datetime import datetime, timedelta
import utils
from tqdm.auto import tqdm
import datasets
import argparse
from itertools import chain
from functools import partial
import numpy as np
from huggingface_hub import login

def get_category_members(category_name, wiki, level=0, max_level=2):
    category = wiki.page(category_name)
    members = category.categorymembers
    articles = []
    progress_bar = tqdm(members.values(), desc=f"Getting articles from {category_name}", disable=level > 0, postfix=f"Articles: {len(articles)}")
    for c in members.values():
        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            articles.extend(get_category_members(c.title, wiki, level=level + 1, max_level=max_level))
        elif c.ns == wikipediaapi.Namespace.MAIN:
            articles.append(c)
        progress_bar.update()
        progress_bar.set_postfix({"Articles": len(articles)})
    return articles

def get_wikipedia_category_events(start_month_year, end_month_year):
    wiki_wiki = wikipediaapi.Wikipedia('DatasetBot/1.0 (wjfu@hust.edu.cn)', 'en')

    # Convert input strings to datetime objects
    start_date = datetime.strptime(start_month_year, '%B %Y')
    end_date = datetime.strptime(end_month_year, '%B %Y')

    current_date = start_date

    events = []

    while current_date <= end_date:
        month_year = current_date.strftime('%B %Y')
        category_name = f"Category:{month_year} events by country"

        articles = get_category_members(category_name, wiki_wiki)
        events.extend(articles)
        
        # Move to the next month
        next_month = current_date.replace(day=28) + timedelta(days=4)  # this will never fail
        current_date = next_month.replace(day=1)

    return events

def extract_event_content(events, contect_fields=["text", "summary"]):
    extracted_events = {field: [] for field in contect_fields}
    for event in tqdm(events, desc="Extracting event content"):
        for field in contect_fields:
            extracted_events[field].append(getattr(event, field))
    extracted_dataset = datasets.Dataset.from_dict(extracted_events)
    return extracted_dataset

def pack_sentence(examples, block_size=128):
    concatenated_examples = " ".join(examples['content'])
    total_words = concatenated_examples.split()
    total_length = len(total_words)
    
    packed_examples = []
    assert total_length >= block_size, "The total length of the examples is less than the block size"
    total_length = total_length - (total_length % block_size)
    for i in range(0, total_length, block_size):
        packed_examples.append(" ".join(total_words[i:i+block_size]))
    return {"content": packed_examples}

def add_label(example, label):
    example["label"] = label
    return example

def add_length(example, block_sizes):
    length = len(example["summary"].split())
    idx = np.searchsorted(block_sizes, length, side="right") - 1
    if idx < 0:
        example["length"] = 0
    else:
        example["length"] = block_sizes[idx]
    truncated_content = " ".join(example["summary"].split()[:block_sizes[idx]])
    example["summary"] = truncated_content
    return example

def main():
    parser = argparse.ArgumentParser(description="Update WikiMIA dataset")
    parser.add_argument("--mem_start", type=str, default="January 2015", help="The start month and year for the member events")
    parser.add_argument("--mem_end", type=str, default="December 2016", help="The end month and year for the member events")
    parser.add_argument("--non_start", type=str, default="March 2024", help="The start month and year for the non-member events")
    parser.add_argument("--non_end", type=str, default="December 2024", help="The end month and year for the non-member events")
    parser.add_argument("--output_dir", type=str, default="WikiMIA-24", help="The output directory to save the dataset")
    parser.add_argument("--block_sizes", type=int, default=[32, 64, 128, 256], nargs="+", help="The block sizes for the dataset")
    parser.add_argument("-t", "--token", type=str, default="your_hftoken")
    
    args = parser.parse_args()
    
    utils.set_proxy()
    login(token=args.token)
    
    mem_start = args.mem_start
    mem_end = args.mem_end
    non_start = args.non_start
    non_end = args.non_end
    
    mem_events = get_wikipedia_category_events(mem_start, mem_end)
    non_events = get_wikipedia_category_events(non_start, non_end)
    mem_raw = extract_event_content(mem_events)
    non_raw = extract_event_content(non_events)
    mem_raw.save_to_disk("WikiMIA-24/mem_raw")
    non_raw.save_to_disk("WikiMIA-24/non_raw")
    # mem_raw = datasets.load_from_disk("WikiMIA-24/mem_raw")
    # non_raw = datasets.load_from_disk("WikiMIA-24/non_raw")
    mem_raw = mem_raw.map(partial(add_label, label=1), load_from_cache_file=False, desc="Adding label to member events")
    non_raw = non_raw.map(partial(add_label, label=0), load_from_cache_file=False, desc="Adding label to non-member events")
    dataset_raw = datasets.concatenate_datasets([mem_raw, non_raw]).map(partial(add_length, block_sizes=args.block_sizes), load_from_cache_file=False, desc="Adding length to the dataset").shuffle()
    dataset_raw = dataset_raw.remove_columns(["text"])
    dataset_raw = dataset_raw.rename_column("summary", "input")
    dataset_pack = {f"WikiMIA_length{block_size}": dataset_raw.filter(lambda example: example["length"] == block_size) for block_size in args.block_sizes}
    
    for split, dataset in dataset_pack.items():
        mem_dataset = dataset.filter(lambda example: example["label"] == 1)
        non_dataset = dataset.filter(lambda example: example["label"] == 0)
        min_length = min(len(mem_dataset), len(non_dataset))
        mem_dataset = mem_dataset.select(range(min_length))
        non_dataset = non_dataset.select(range(min_length))
        dataset_pack[split] = datasets.concatenate_datasets([mem_dataset, non_dataset]).shuffle().remove_columns(["length"])
        
    dataset_pack = datasets.DatasetDict(dataset_pack)
    dataset_pack.save_to_disk(args.output_dir)
    dataset_pack.push_to_hub("wjfu99/WikiMIA-24", private=False)
    

if __name__ == "__main__":
    main()