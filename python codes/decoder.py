# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time

# Importing the tokenizer
from basictokenizer import BasicTokenizer

# # hyperparameters
batch_size = 64 
block_size = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"The device is {device}")
eval_iters = 200
n_embd = 232
n_head = 6
n_layer = 5
dropout = 0.2
vocab_size = 500

# Opening the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# In corporate the tokenization
tokenize = BasicTokenizer()

# Train and val splits
data = torch.tensor(tokenize.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y with random off sets
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    # Calculate the loss every 500 iters avaraging it over last 200 iters
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Calculate the model size
def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size

class Head(nn.Module):
    # One head of Self Attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x --> (B , T , hs)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Communication
        self.ffwd = FeedFoward(n_embd) # Calculation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) -> Broadcasting
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx (B , T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # cropping context
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = Decoder()
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Getting the activations and gradients of ReLU for plotting
relu_outputs = {}
relu_gradients = {}
def forward_hook(name):
    def hook(module, input, output):
        relu_outputs[name] = output.detach()
    return hook
def backward_hook(name):
    def hook(module, grad_input, grad_output):
        relu_gradients[name] = grad_output[0].detach()
    return hook
def get_relu_stats():
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(forward_hook(name))
            module.register_backward_hook(backward_hook(name))
get_relu_stats()

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_t = []
loss_v = []
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        loss_t.append(losses["train"])
        loss_v.append(losses["val"])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Plotting the training and validation loss
training_loss = [tensor.item() for tensor in loss_t]
val_loss = [tensor.item() for tensor in loss_v]
plt.plot(training_loss, color='blue', label='Training Loss')
plt.plot(val_loss, color='red', label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.savefig("Loss Plot")
# plt.show()


# Plotting the ReLU outputs
plt.figure(figsize=(10, 6))
for key, tensor in relu_outputs.items():
    flattened_tensor = tensor.flatten()
    mean = flattened_tensor.mean().item()
    std = flattened_tensor.std().item()
    percentage_le_zero = (flattened_tensor <= 0.0).sum().item() / flattened_tensor.numel() * 100 # satuarted ReLU %
    # print(f'{key}: Mean = {mean:.2f}, Std = {std:.2f}, Percentage <= 0.0 = {percentage_le_zero:.2f}%')
    plt.hist(flattened_tensor.cpu().numpy(), bins=50, alpha=0.5, label=f'{key} (Mean: {mean:.2f}, Std: {std:.2f}, <=0: {percentage_le_zero:.2f}%)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('ReLU activations')
plt.legend()
plt.savefig("ReLU Activations")
# plt.show()

# Plotting the ReLU gradients
plt.figure(figsize=(10, 6))
for key, tensor in relu_gradients.items():
    flattened_tensor = tensor.flatten()
    mean = flattened_tensor.mean().item()
    std = flattened_tensor.std().item()
    # print(f'{key}: Mean = {mean:.2f}, Std = {std:.2f}')
    plt.hist(flattened_tensor.cpu().numpy(), bins=50, alpha=0.5, label=f'{key} (Mean: {mean:.10f}, Std: {std:.10f}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('ReLU Gradients')
plt.legend()
plt.savefig("ReLU gradients")
plt.show()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
start = time.time()
# print(tokenize.decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
open('output.txt', 'w').write(tokenize.decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
end = time.time()
print(f"Time taken to generate 10,000 tokens with the unquantised model is {(end-start)/60}")
time_nq = (end-start)/60