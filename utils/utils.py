import torch
import torch.nn.functional as F

def generate_text(
    model,
    tokenizer,
    text,
    max_new_tokens,
    context_size,
    device=None,
    temperature=1.4,
    top_k=25,
    eos_id=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    token_ids = tokenizer.encode(text)
    encoded_tokens = torch.tensor(token_ids, device=device).unsqueeze(
        0
    )  # Ensure tensor is on the correct device

    for _ in range(max_new_tokens):
        current_logits_context = encoded_tokens[:, -context_size:]

        with torch.no_grad():
            logits = model(current_logits_context)
        logits = logits[:, -1, :]
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # Sample from the distribution
            token_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            token_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if token_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        encoded_tokens = torch.cat((encoded_tokens, token_next), dim=1)

    encoded_tokens = encoded_tokens.squeeze(0)
    decoded = tokenizer.decode(encoded_tokens.tolist())

    return decoded

def validate(model, test_dataloader, device, context_length, tokenizer):
  
    model.eval() 
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for global_step, (input_batch, output_batch) in enumerate(test_dataloader):
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            
            logits = model(input_batch)
            final_logits = logits[:,-1,:]
            loss = F.cross_entropy(final_logits, output_batch.squeeze(dim=-1))
            
            total_loss += loss.item()
            total_tokens += input_batch.numel()
    
            if global_step % 100 == 0:
                print(f"Validation Loss - {loss.item()}")
                print(f"Tokens Processed - {total_tokens}")
                text_list = generate_text(model, tokenizer=tokenizer, text="The quick brown f", max_new_tokens=15, context_size=context_length)
                print(text_list.replace("\n", " "))

    average_loss = total_loss / (global_step+1)
    return average_loss, total_tokens, text_list
