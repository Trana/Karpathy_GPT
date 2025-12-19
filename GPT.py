import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iterations = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_vector_size = 384
num_heads = 6
num_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
chars_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading returns in format
# input is [25] the target: 17
# input is [25, 17] the target: 27
# input is [25, 17, 27] the target: 10
# input is [25, 17, 27, 10] the target: 0
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# this tells pytorch to not store values for backprop
@torch.no_grad()
def estimate_loss():
    out = {}
    # .eval and .train does not do anything for this model, but good practice
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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_vector_size, head_size, bias=False)
        self.query = nn.Linear(embedding_vector_size, head_size, bias=False)
        self.value = nn.Linear(embedding_vector_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head size)
        q = self.query(x) # (B,T,head size)
        # compute attention scores ("affinities")
        # Transpose to swal the inner dimensions so that we can matrix multiplication
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_vector_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embedding_vector_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_vector_size, 4 * embedding_vector_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_vector_size, embedding_vector_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embedding_vector_size, num_heads):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_vector_size // num_heads
        # Double // rounds down to closes integer
        self.sa_heads = MultiHeadAttention(num_heads, head_size) # 4 heads of self-attention 8-dimensional each. Contactenated to return 32-dimensional
        self.feed_forward = FeedFoward(embedding_vector_size)
        self.ln1 = nn.LayerNorm(embedding_vector_size)
        self.ln2 = nn.LayerNorm(embedding_vector_size)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(chars_size, embedding_vector_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_vector_size)
        
        self.blocks = nn.Sequential(*[Block(embedding_vector_size, num_heads=num_heads) for _ in range(num_layer)]) # 4 layers of transformer blocks
        self.layer_normalizer = nn.LayerNorm(embedding_vector_size)
        
        # language model head, input embedding and output logit scores for each character
        self.lm_head = nn.Linear(embedding_vector_size, chars_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # This does a lookup of every char token in the embedding table and replaces it with the corresponding embedding vector
        embedding_for_token = self.token_embedding_table(idx) # (B,T,C)
        embedding_for_position = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # Pytorch just adds and empty dimension to the position embedding to make the shapes match
        x = embedding_for_token + embedding_for_position # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        # Logits are the scores for each character in the vocabulary
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape the logits and targets so that we can compute the cross-entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # compute the loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iterations):

    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))