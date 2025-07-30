import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.distributions as td

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
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, N, start_token=2, dropout=0.1):
        super(AutoregressiveTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.N = N

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.angle_encoder = nn.Linear(N, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length+1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 2)

    def forward(self, src, angles):
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        angle_embed = self.angle_encoder(angles)
        angle_embed = torch.reshape(angle_embed, (angle_embed.size(0),1,angle_embed.size(1)))
        angle_embed = torch.tile(angle_embed, (1,src.size(1),1))
        embedded = torch.cat((angle_embed,embedded),axis=0)
        embedded = self.pos_encoder(embedded)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded.size(0)).to(src.device)
        output = self.transformer_encoder(embedded, mask=tgt_mask)

        logits = self.fc_out(output)
        return logits

    def generate_and_log_prob(self, angles, batch_size, device):
        generated = torch.full((1, batch_size), self.start_token, dtype=torch.long, device=device)
        log_probs = []
        for _ in range(self.max_seq_length - 1):
            logits = self.forward(generated, angles)
            next_token_logits = logits[-1, :, :]
            probs = F.softmax(next_token_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample().unsqueeze(0)
            log_prob = dist.log_prob(next_token.squeeze(0))
            log_probs.append(log_prob)
            generated = torch.cat((generated, next_token), dim=0)
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



def build_subspace_hamiltonian(batch, angles, J, h):
    batch = torch.unique(batch, dim=0).detach()
    B, N = batch.shape
    device = batch.device

    rotated_Z = torch.empty((N, 2, 2), dtype=torch.float64, device=device)
    rotated_X = torch.empty((N, 2, 2), dtype=torch.float64, device=device)

    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.float64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64, device=device)

    for i in range(N):
        theta = angles[i]
        cos_theta = torch.cos(theta / 2)
        sin_theta = torch.sin(theta / 2)

        U = torch.zeros(2, 2, dtype=torch.float64, device=device)
        U[0, 0] = cos_theta
        U[0, 1] = -sin_theta
        U[1, 0] = sin_theta
        U[1, 1] = cos_theta

        rotated_Z[i] = U.conj().T @ Z @ U
        rotated_X[i] = U.conj().T @ X @ U

    eq_all = (batch.unsqueeze(1) == batch.unsqueeze(0))

    H = torch.zeros((B, B), dtype=torch.float64, device=device)

    for site in range(N - 1):
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[site] = mask[site + 1] = False
        condition = eq_all[..., mask].all(dim=-1)

        x_site = batch[:, site].unsqueeze(1)
        y_site = batch[:, site].unsqueeze(0)
        term1 = rotated_Z[site][x_site, y_site]

        x_site_next = batch[:, site + 1].unsqueeze(1)
        y_site_next = batch[:, site + 1].unsqueeze(0)
        term2 = rotated_Z[site + 1][x_site_next, y_site_next]

        H = H + (-J[site]) * condition * (term1 * term2)

    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[0] = mask[N - 1] = False
    condition = eq_all[..., mask].all(dim=-1)
    x_last = batch[:, N - 1].unsqueeze(1)
    y_last = batch[:, N - 1].unsqueeze(0)
    term_last = rotated_Z[N - 1][x_last, y_last]

    x_first = batch[:, 0].unsqueeze(1)
    y_first = batch[:, 0].unsqueeze(0)
    term_first = rotated_Z[0][x_first, y_first]
    H = H + (-J[-1]) * condition * (term_last * term_first)

    for site in range(N):
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[site] = False
        condition = eq_all[..., mask].all(dim=-1)
        x_site = batch[:, site].unsqueeze(1)
        y_site = batch[:, site].unsqueeze(0)
        term = rotated_X[site][x_site, y_site]
        H = H + (-h) * condition * term

    eigvals, _ = torch.linalg.eigh(H)
    return eigvals[0]



N = 6
J = np.ones(N)
h = 0.5

# hyperpatameters of the Transformer
d_model = 64
nhead = 4
num_layers = 2
max_seq_length = N+1
vocab_size = 3  # tokens {0,1, start=2}
batch_size = 32
num_batches = 4
num_steps = 500 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

angles = torch.randn(N, dtype=torch.float64, device=device)
angles.requires_grad = True

transformer = AutoregressiveTransformer(vocab_size=vocab_size,d_model=d_model,nhead=nhead,num_layers=num_layers,
                                        max_seq_length=max_seq_length,N=N,start_token=2,dropout=0.1).to(device)

optimizer_transformer = optim.Adam(transformer.parameters(), lr=1e-3)
optimizer_angles = optim.Adam([angles], lr=1e-2)

for step in range(num_steps):
    all_energies = []
    all_energies_grad = []
    all_logps = []

    for _ in range(num_batches):

        config, logp_transformer = transformer.generate_and_log_prob(
            angles.unsqueeze(0).float(), batch_size, device
        )

        E_batch = build_subspace_hamiltonian(config, angles, J=J, h=h)
        E_batch = E_batch/N
        all_energies_grad.append(E_batch)
        all_energies.append(E_batch.detach().item())
        all_logps.append(logp_transformer.sum()) 

    baseline = float(np.mean(all_energies))


    loss = torch.tensor(0.0, device=device)
    for i,_ in enumerate((all_energies)):
        loss = loss + ((all_energies[i] - baseline) * all_logps[i]) / num_batches + (all_energies_grad[i] - baseline) / num_batches

    optimizer_angles.zero_grad()
    optimizer_transformer.zero_grad()

    loss.backward()

    optimizer_angles.step()
    optimizer_transformer.step()

    print(f"Step {step}, Loss: {loss.item()}, Energy: {np.min(all_energies)}, Baseline: {baseline}")



