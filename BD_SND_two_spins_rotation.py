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
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, N, start_token=2, dropout=0.1,trotter_steps=2):
        super(AutoregressiveTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.N = N
        self.trotter_steps = trotter_steps

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.angle_encoder = nn.Linear((1 + 2*self.trotter_steps), d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
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


def build_subspace_hamiltonian(batch, angles, trotter_steps, J=1.0, h=1.0):
    batch = torch.unique(batch, dim=0).detach()
    B, N = batch.shape
    assert N % 2 == 0, "Even N"
    device = batch.device

    P = N // 2  
    block_angles_y = angles[0].view(1)
    block_angles_zz = angles[1:1+trotter_steps].view(trotter_steps)
    block_angles_x = angles[1+trotter_steps:].view(trotter_steps)

    I2 = torch.eye(2, dtype=torch.complex128, device=device)
    Z  = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    X  = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    
    ZZ_rot = torch.empty((P, 4, 4), dtype=torch.complex128, device=device)
    ZI_rot = torch.empty((P, 4, 4), dtype=torch.complex128, device=device)
    IZ_rot = torch.empty((P, 4, 4), dtype=torch.complex128, device=device)
    XI_rot = torch.empty((P, 4, 4), dtype=torch.complex128, device=device)
    IX_rot = torch.empty((P, 4, 4), dtype=torch.complex128, device=device)

    for p in range(P):
        alfa0 = block_angles_y
        alfa1 = block_angles_y
        cy0 = torch.cos(alfa0 / 2)
        sy0 = torch.sin(alfa0 / 2)
        Ry0 = torch.zeros(2, 2, dtype=torch.complex128, device=device)
        Ry0[0, 0] = cy0
        Ry0[0, 1] = -sy0
        Ry0[1, 0] = sy0
        Ry0[1, 1] = cy0

        cy1 = torch.cos(alfa1 / 2)
        sy1 = torch.sin(alfa1 / 2)
        Ry1 = torch.zeros(2, 2, dtype=torch.complex128, device=device)
        Ry1[0, 0] = cy1
        Ry1[0, 1] = -sy1
        Ry1[1, 0] = sy1
        Ry1[1, 1] = cy1
        Ry_joint = torch.kron(Ry0, Ry1)
        U_block = Ry_joint

        for t in range(trotter_steps):
            gamma = block_angles_zz[t] 
            beta0 = block_angles_x[t]
            beta1 = block_angles_x[t]

            exp_neg_ig_half = torch.exp(-0.5j * gamma)
            exp_pos_ig_half = torch.exp(0.5j * gamma)

            Rzz_block = torch.zeros(4, 4, dtype=torch.complex128, device=device)
            Rzz_block[0, 0] = exp_neg_ig_half
            Rzz_block[1, 1] = exp_pos_ig_half
            Rzz_block[2, 2] = exp_pos_ig_half
            Rzz_block[3, 3] = exp_neg_ig_half

            cy0 = torch.cos(beta0 / 2)
            sy0 = torch.sin(beta0 / 2)
            Rx0 = torch.zeros(2, 2, dtype=torch.complex128, device=device)
            Rx0[0, 0] = cy0
            Rx0[0, 1] = -1j*sy0
            Rx0[1, 0] = -1j*sy0
            Rx0[1, 1] = cy0

            cy1 = torch.cos(beta1 / 2)
            sy1 = torch.sin(beta1 / 2)
            Rx1 = torch.zeros(2, 2, dtype=torch.complex128, device=device)
            Rx1[0, 0] = cy1
            Rx1[0, 1] = -1j*sy1
            Rx1[1, 0] = -1j*sy1
            Rx1[1, 1] = cy1
            Rx_joint = torch.kron(Rx0, Rx1)

            layer_U = Rx_joint @ Rzz_block
            U_block = layer_U @ U_block 

        ZZ = torch.kron(Z, Z)
        ZI = torch.kron(Z, I2)
        IZ = torch.kron(I2, Z)
        XI = torch.kron(X, I2)
        IX = torch.kron(I2, X)

        U_dagger = U_block.conj().T
        ZZ_rot[p] = U_dagger @ ZZ @ U_block
        ZI_rot[p] = U_dagger @ ZI @ U_block
        IZ_rot[p] = U_dagger @ IZ @ U_block
        XI_rot[p] = U_dagger @ XI @ U_block
        IX_rot[p] = U_dagger @ IX @ U_block

    eq_all = (batch.unsqueeze(1) == batch.unsqueeze(0))  
    H = torch.zeros((B, B), dtype=torch.complex128, device=device)

    for p in range(P):
        i0, i1 = 2*p, 2*p + 1
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[i0] = mask[i1] = False
        cond = eq_all[..., mask].all(dim=-1)  

        idx_bra_configs = 2*batch[:, i0] + batch[:, i1]
        idx_ket_configs = idx_bra_configs
        term = ZZ_rot[p][idx_bra_configs.unsqueeze(1), idx_ket_configs.unsqueeze(0)]
        H = H + (-J) * cond * term

    for p_idx in range(P):
        q_idx = (p_idx + 1) % P

        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[2*p_idx] = mask[2*p_idx+1] = False
        mask[2*q_idx] = mask[2*q_idx+1] = False
        cond = eq_all[..., mask].all(dim=-1)

        idx_bra_p = 2*batch[:, 2*p_idx] + batch[:, 2*p_idx+1]
        idx_ket_p = idx_bra_p
        term_p = IZ_rot[p_idx][idx_bra_p.unsqueeze(1), idx_ket_p.unsqueeze(0)]

        idx_bra_q = 2*batch[:, 2*q_idx] + batch[:, 2*q_idx+1]
        idx_ket_q = idx_bra_q
        term_q = ZI_rot[q_idx][idx_bra_q.unsqueeze(1), idx_ket_q.unsqueeze(0)]

        H = H + (-J) * cond * (term_p * term_q)

    for p in range(P):
        i0, i1 = 2*p, 2*p + 1
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[i0] = mask[i1] = False
        cond = eq_all[..., mask].all(dim=-1)

        idx_bra_configs = 2*batch[:, i0] + batch[:, i1]
        idx_ket_configs = idx_bra_configs

        term_Xi = XI_rot[p][idx_bra_configs.unsqueeze(1), idx_ket_configs.unsqueeze(0)]
        term_Xj = IX_rot[p][idx_bra_configs.unsqueeze(1), idx_ket_configs.unsqueeze(0)]

        H = H + (-h) * cond * (term_Xi + term_Xj)

    eigvals, _ = torch.linalg.eigh(H)
    
    return eigvals[0]





h = 0.5
N = 6
trotter_steps = 1
J = 1

d_model = 64
nhead = 4
num_layers = 2
max_seq_length = N + 1
vocab_size = 3
num_batches = 4
num_steps = 500
batch_size = 32


device = 'cuda' if torch.cuda.is_available() else 'cpu'

angles = torch.randn(1+2*trotter_steps, dtype=torch.float64, device=device)
angles.requires_grad = True

transformer = AutoregressiveTransformer(vocab_size, d_model, nhead, num_layers, max_seq_length, N,trotter_steps=trotter_steps).to(device)
optimizer_transformer = optim.Adam(transformer.parameters(), lr=1e-3)
optimizer_angles = optim.Adam([angles], lr=1e-2)




for step in range(num_steps):
    optimizer_transformer.zero_grad()
    optimizer_angles.zero_grad()
    dummy_input = torch.zeros(1, 1).to(device)

    en = []
    en_grad = []
    log_prob_transformer = []
    for batches in range(num_batches):
        angles0 = angles.unsqueeze(0).expand(1,1+2*trotter_steps).float()
        configuration, log_prob_transformer0 = transformer.generate_and_log_prob(angles0, batch_size, device)
        log_prob_transformer.append(log_prob_transformer0.sum(dim=0))

        E_batch = build_subspace_hamiltonian(configuration,
                                                    angles, trotter_steps, J=1.0, h=h) 
        E_batch = E_batch / N
        en.append(E_batch.detach().item())
        en_grad.append(E_batch)
    baseline = np.mean(en)
    
    loss = torch.tensor(0.0, device=device)
    for i,_ in enumerate((en)):
        loss = loss + ((en[i] - baseline) * log_prob_transformer[i]) / num_batches + (en_grad[i] - baseline) / num_batches

    optimizer_angles.zero_grad()
    optimizer_transformer.zero_grad()

    loss.backward()
    
    optimizer_angles.step()
    optimizer_transformer.step()

    print(f"Step {step:4d}, Loss: {loss.item():.4f}, Energy: {np.min(en):.4f}, Baseline: {baseline:.4f}")

