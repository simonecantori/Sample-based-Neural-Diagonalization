import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, start_token=2, dropout=0.1):
        super(AutoregressiveTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.max_seq_length = max_seq_length 
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 2)
        
    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        seq_len = src.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(embedded, mask=tgt_mask)

        logits = self.fc_out(output)
        return logits

    def generate_and_log_prob(self, batch_size, device):
        generated = torch.full((1, batch_size), self.start_token, dtype=torch.long, device=device)
        log_probs = []
        for _ in range(self.max_seq_length - 1):
            logits = self.forward(generated)
            next_token_logits = logits[-1, :, :] 
            probs = F.softmax(next_token_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)
            log_probs.append(log_prob)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=0)

        # Collecting unique configurations
        configuration = generated[1:].transpose(0, 1)
        total_log_prob = torch.stack(log_probs, dim=0)
        unique, idx, counts = torch.unique(configuration, dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0],device=device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        total_log_prob=total_log_prob[:,first_indicies]
        total_log_prob = total_log_prob.sum(dim=0)
        return unique, total_log_prob

def build_subspace_hamiltonian(batch, J, h):
    N = batch.shape[1]

    batch = batch.detach()
    batch = torch.unique(batch, dim=0)
    batch_size = batch.shape[0]
    H = torch.zeros((batch_size, batch_size),device=device)
    spins = 1 - 2 * batch
    
    for i in range(batch_size):
        diag_energy = torch.sum(-J[:-1] * spins[i, :-1] * spins[i, 1:]) - J[-1] * spins[i, -1] * spins[i, 0]
        H[i, i] = diag_energy
    for site in range(N):
        flipped = batch.clone()
        flipped[:, site] = 1 - flipped[:, site]
        matches = (flipped.unsqueeze(1) == batch.unsqueeze(0)).all(dim=2)
        H[matches] -= h

    eigvals = torch.linalg.eigh(H)[0]
    return eigvals[0]


N = 6  # number of spins
h = 0.5 # Magnetic field

# hyperpatameters of the Transformer
d_model = 64 
nhead = 4
num_layers = 2

max_seq_length = N + 1 # The sequence length is N+1 (one start token + N spins).
vocab_size = 3  # tokens: 0, 1, and the start token (2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoregressiveTransformer(vocab_size, d_model, nhead, num_layers, max_seq_length, start_token=2, dropout=0.1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_steps = 100
batch_size = 32 # Number of non-unique configurations sampled for each batch 
num_batches = 16

J = torch.ones(N, device=device)

for step in range(num_steps):
    optimizer.zero_grad()
    loss = 0.
    en = []
    log_prob_transformer = []
    for K in range(num_batches):
        configuration, log_prob = model.generate_and_log_prob(batch_size, device)
        log_prob_transformer.append(log_prob.sum(dim=0))

        E_batch = build_subspace_hamiltonian(configuration, J=J, h=h) / N
        en.append(E_batch)
    en = torch.tensor(en)
    baseline = torch.mean(en)
    loss = ((en[0] - baseline) * (log_prob_transformer[0]))/num_batches
    for batches in range(1, num_batches):
        loss += ((en[batches] - baseline) * (log_prob_transformer[batches]))/num_batches

    loss.backward()
    optimizer.step()
    
    print(f"Step {step}, Loss: {loss.item()}, Lowest Energy: {torch.min(en)}, Baseline: {baseline}")

