import torch.nn as nn
from torch.nn import functional as F
import torch
import time
from model.model import Model
from data.dataset import BookCorpusDataset, create_dataloader, return_book_corpus_text
import tiktoken
import yaml
from utils.utils import generate_text, validate
from torch.optim.lr_scheduler import StepLR
import os
from model.tokenizer import tokenizer_v1

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

tokenizer = tokenizer_v1()
print(f"Fetching Text data")

save_folder = "pretrain_data"
# files_wikitext = return_book_corpus_text(save_folder)
print(f"Fetched Text data.")

vocab_size = config["model"]["vocab_size"]
context_length = config["model"]["context_length"]
emb_dim = config["model"]["emb_dim"]
n_heads = config["model"]["n_heads"]
n_layers = config["model"]["n_layers"]
drop_rate = config["model"]["drop_rate"]
qkv_bias = config["model"]["qkv_bias"]

stride = config["data"]["stride"]
batch_size = config["data"]["batch_size"]

file_paths = os.listdir(save_folder)
train_paths = file_paths[:-2]
test_paths = file_paths[-2:]

print("Building datasets.")
train_dataset = BookCorpusDataset(
    save_folder, train_paths, tokenizer, context_length, stride
)
test_dataset = BookCorpusDataset(
    save_folder, test_paths, tokenizer, context_length, stride
)

print("creating dataloaders.")
train_dataloader = create_dataloader(
    train_dataset, batch_size=batch_size, shuffle=False
)
test_dataloader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Loading the Model.")
model = Model(
    vocab_size=vocab_size,
    emb_dim=emb_dim,
    context_length=context_length,
    drop_rate=drop_rate,
    n_layers=n_layers,
    n_heads=n_heads,
    qkv_bias=qkv_bias,
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,}")
print(f"Loaded the Model.")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)

total_tokens = 0

model.train()
for global_step, (input_batch, output_batch) in enumerate(train_dataloader, 1):
    input_batch, output_batch = input_batch.to(device), output_batch.to(device)

    optimizer.zero_grad()
    logits = model(input_batch)
    final_logits = logits[:, -1, :]
    loss = F.cross_entropy(final_logits, output_batch.squeeze(dim=-1))

    loss.backward()
    optimizer.step()

    total_tokens += input_batch.numel()

    if (global_step) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = F.cross_entropy(final_logits, output_batch.squeeze(dim=-1))
            print(f"Train Loss - {train_loss}")
            print(f"Tokens Processed - {total_tokens}")
            text_list = generate_text(
                model,
                tokenizer=tokenizer,
                text="Coun",
                max_new_tokens=12,
                context_size=context_length,
            )
            print(text_list.replace("\n", " "))
            if global_step % 1000 == 0:
                test_loss, test_tokens, test_text = validate(
                    model, test_dataloader, device, context_length, tokenizer
                )
                torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            f"checkpoint-model_{test_loss}.pth",
            )       
            else:
                test_loss, test_tokens, test_text = "", "", ""
            model.train()
            with open("log.txt", "a") as file:
                file.writelines(
                    [
                        f"Train Loss-{train_loss}\n",
                        f"Tokens Processed-{total_tokens}\n",
                        f"{text_list}\n",
                        f"Test Loss-{test_loss}\n",
                        f"Test Tokens-{test_tokens}\n",
                        f"{test_text}\n" f"---------------------------\n",
                    ]
                )
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    "checkpoint-model_final.pth",
)
