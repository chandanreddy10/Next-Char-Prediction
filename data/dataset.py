from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import os
import torch
import ast
import tiktoken
import re

class BookCorpusDataset(IterableDataset):
    def __init__(self, root_path, file_paths, tokenizer, context_length, stride):
        self.root_path = root_path
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride

    def __iter__(self):
        for file_path in self.file_paths:
            file_path = os.path.join(self.root_path, file_path)
            with open(file_path, "r") as file:
                text = file.read()
            token_ids = self.tokenizer.encode(text)

            for index in range(0, len(token_ids) - self.context_length, self.stride):
                # input_text = self.tokenizer.decode(token_ids[index:index+self.context_length])
                input_ids = torch.tensor(token_ids[index : index + self.context_length])
                target_ids = torch.tensor(
                    token_ids[index+self.context_length:index+self.context_length+1]
                )
                # target_text = self.tokenizer.decode(token_ids[index+self.context_length:index+self.context_length+1])
                yield input_ids, target_ids


def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    num_workers=4,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def format_for_pretraining(data):
    # Extract title, text, and metadata
    if isinstance(data["METADATA"], str):
        metadata = ast.literal_eval(data["METADATA"])
    else:
        metadata = data["METADATA"]
    title = metadata["title"] if "METADATA" in data and "title" in metadata else ""
    text = data["TEXT"] if "TEXT" in data else ""

    title = title.replace("\r", "")
    text = text.replace("\r", "")
    text = text.replace("\n", "")

    formatted_output = f"{title}\n{text}"

    return formatted_output

def return_book_corpus_text(
    root_folder="pretrain_data", max_chars=2000000000, max_file_size=10000000
):
    os.makedirs(root_folder, exist_ok=True)
    # DATASET_ID = "Salesforce/wikitext"
    DATASET_ID = "legacy-datasets/wikipedia"
    final_characters = 0
    track_characters = 0

    store_files = []
    gutenberg = load_dataset(DATASET_ID,"20220301.en", split="train", streaming=True,trust_remote_code=True)
    for index, element in enumerate(gutenberg):
        formatted_element = element['text']
        formatted_element = re.sub(r"[^a-zA-Z\s.,!?;:()'\"-]", "", formatted_element)
        formatted_element = re.sub(r"\s+", " ", formatted_element).strip()
        chars = len(formatted_element)
        final_characters += chars

        track_characters += chars
        store_files.append(formatted_element)

        if track_characters >= max_file_size:
            with open(f"{root_folder}/{index}_file.txt", "w") as file:
                file.write("".join(store_files))
            store_files = []
            track_characters = 0
            print(f"Writing {root_folder}/{index}_file.txt")
            print(final_characters)
        if final_characters > max_chars:
            break
        if index>8000:
            break
