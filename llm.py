import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
assert device == 'cuda', "This code is not optimized for CPU"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Head(nn.Module):
    def __init__(self, C, H, T):
        super().__init__()
        self.query = nn.Linear(C, H, bias=False)
        self.key = nn.Linear(C, H, bias=False)
        self.value = nn.Linear(C, H, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        self.H = H
        self.T = T

    def forward(self, x):
        query_vectors = self.query(x)
        key_vectors = self.key(x)

        tril = self.tril
        wei = torch.zeros(self.T, self.T, device=x.device)
        wei = wei.masked_fill(tril == 0, float('-inf'))

        attention_pattern = query_vectors @ key_vectors.transpose(-2, -1)
        attention_pattern = attention_pattern / (self.H ** 0.5)
        attention_weights = F.softmax(attention_pattern + wei, dim=-1)

        value_vectors = self.value(x)
        context = attention_weights @ value_vectors

        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, H, C, n_heads, T):
        super().__init__()
        self.heads = nn.ModuleList([Head(C, H, T) for _ in range(n_heads)])
        self.combine_heads = nn.Linear(H*n_heads, C)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.combine_heads(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, C, feedforward_factor=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, C * feedforward_factor),
            nn.ReLU(),
            nn.Linear(C * feedforward_factor, C),
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, C, use_affine=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(C)) if use_affine else None
        self.beta = nn.Parameter(torch.zeros(C)) if use_affine else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if self.gamma is not None and self.beta is not None:
            return self.gamma * (x - mean) / (std + 1e-6) + self.beta
        else:
            return (x - mean) / (std + 1e-6)

class Block(nn.Module):
    def __init__(self, H, C, n_heads, T):
        super().__init__()
        self.attention = MultiHeadAttention(H, C, n_heads, T)
        self.ff = FeedForward(C)
        self.norm1 = LayerNorm(C, use_affine=True)
        self.norm2 = LayerNorm(C, use_affine=True)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class BottleNeckGPT(nn.Module):
    def __init__(
            self,
            B,
            T,
            C,
            n_heads,
            H,
            n_layers,
            vocab_size,
            bottleneck_model,
            bottleneck_location
        ):
        super().__init__()
        self.B = B
        self.T = T
        self.C = C
        self.n_heads = n_heads
        self.H = H
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(T, C)
        self.lm_head = nn.Linear(C, vocab_size)
        self.layers = nn.ModuleList([Block(H, C, n_heads, T) for _ in range(n_layers)])
        self.bottleneck_model = bottleneck_model
        self.bottleneck_location = bottleneck_location
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None, bottleneck_early_stop=False):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        results = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.bottleneck_location:
                if self.bottleneck_model:
                    bm_results = self.bottleneck_model(x)
                    x = bm_results['decoded']
                    results['bm_results'] = bm_results
                    if bottleneck_early_stop:
                        return results

        logits = self.lm_head(x)
        results['logits'] = logits

        if targets is not None:
            logits_loss_view = logits.view(-1, self.vocab_size)
            targets_loss_view = targets.view(-1)
            loss = F.cross_entropy(logits_loss_view, targets_loss_view)
            results['loss'] = loss

        return results

    def generate(self, idx, max_new_tokens, temperature=0.5):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.T:])['logits']
            last_token_logits = logits[:, -1, :]
            last_token_logits = last_token_logits / temperature
            probabilities = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def prompt_model(self, prompt, max_new_tokens, encode_fn, decode_fn, bottleneck_model=None, temperature=0.5):
        autoregressive_seq = encode_fn(prompt)
        for _ in range(max_new_tokens):
            prediction_index = len(autoregressive_seq) - 1

            model_input = torch.tensor(autoregressive_seq, device=self.token_embedding_table.weight.device)

            while model_input.shape[0] < self.T:
                pad_token = torch.tensor(encode_fn("\n"), device=self.token_embedding_table.weight.device)
                model_input = torch.cat((model_input, pad_token), dim=0)

            model_input = model_input.unsqueeze(0)

            model_output = self(model_input, bottleneck_model=bottleneck_model)
            logits = model_output['logits']
            prediction_token = logits[:, prediction_index, :] / temperature
            probabilities = F.softmax(prediction_token, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = next_token.item()

            autoregressive_seq.append(next_token)

        return decode_fn(autoregressive_seq)

    def get_embedding(self, prompt, encode_fn, model_embedding_layer):
        sequence = encode_fn(prompt)
        model_input = torch.tensor(sequence, device=self.token_embedding_table.weight.device)
        sequence_index = len(sequence) - 1
        while model_input.shape[0] < self.T:
            pad_token = torch.tensor(encode_fn("\n"), device=self.token_embedding_table.weight.device)
            model_input = torch.cat((model_input, pad_token), dim=0)
        model_input = model_input.unsqueeze(0)
        embedding = self.forward(model_input)['logits']
        embedding = embedding.squeeze(0)[sequence_index]
        return embedding

class GPT(nn.Module):
    def __init__(
            self,
            B,
            T,
            C,
            n_heads,
            H,
            n_layers,
            vocab_size,
        ):
        super().__init__()
        self.B = B
        self.T = T
        self.C = C
        self.n_heads = n_heads
        self.H = H
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(T, C)
        self.lm_head = nn.Linear(C, vocab_size)
        self.layers = nn.ModuleList([Block(H, C, n_heads, T) for _ in range(n_layers)])
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None, stop_at_layer=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        results = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == stop_at_layer:
                results['residuals'] = x
                return results

        logits = self.lm_head(x)
        results['logits'] = logits

        if targets is not None:
            logits_loss_view = logits.view(-1, self.vocab_size)
            targets_loss_view = targets.view(-1)
            loss = F.cross_entropy(logits_loss_view, targets_loss_view)
            results['loss'] = loss

        return results

    def generate(self, idx, max_new_tokens, temperature=0.5):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.T:])['logits']
            last_token_logits = logits[:, -1, :]
            last_token_logits = last_token_logits / temperature
            probabilities = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def prompt_model(self, prompt, max_new_tokens, encode_fn, decode_fn, bottleneck_model=None, temperature=0.5):
        autoregressive_seq = encode_fn(prompt)
        for _ in range(max_new_tokens):
            prediction_index = len(autoregressive_seq) - 1

            model_input = torch.tensor(autoregressive_seq, device=self.token_embedding_table.weight.device)

            while model_input.shape[0] < self.T:
                pad_token = torch.tensor(encode_fn("\n"), device=self.token_embedding_table.weight.device)
                model_input = torch.cat((model_input, pad_token), dim=0)

            model_input = model_input.unsqueeze(0)

            model_output = self(model_input, bottleneck_model=bottleneck_model)
            logits = model_output['logits']
            prediction_token = logits[:, prediction_index, :] / temperature
            probabilities = F.softmax(prediction_token, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = next_token.item()

            autoregressive_seq.append(next_token)

        return decode_fn(autoregressive_seq)

    def get_embedding(self, prompt, encode_fn, model_embedding_layer):
        sequence = encode_fn(prompt)
        model_input = torch.tensor(sequence, device=self.token_embedding_table.weight.device)
        sequence_index = len(sequence) - 1
        while model_input.shape[0] < self.T:
            pad_token = torch.tensor(encode_fn("\n"), device=self.token_embedding_table.weight.device)
            model_input = torch.cat((model_input, pad_token), dim=0)
        model_input = model_input.unsqueeze(0)
        embedding = self.forward(model_input)['logits']
        embedding = embedding.squeeze(0)[sequence_index]
        return embedding