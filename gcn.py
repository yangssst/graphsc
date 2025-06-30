import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if isinstance(adj, torch.sparse.FloatTensor):
            out = torch.sparse.mm(adj, support)
        else:
            out = torch.matmul(adj, support)
        return out + self.bias

def attn_head(seq, out_sz, adj_mat, activation, in_drop=0.0, coef_drop=0.5, residual=False):
    N, D = seq.shape
    device = seq.device

    # Dropout
    if in_drop != 0.0:
        seq = F.dropout(seq, p=in_drop, training=True)

    # [N,D] -> [N,D']
    seq_fts = F.linear(seq, torch.empty(out_sz, D, device=device).normal_())

    # [N,1]
    f_1 = F.linear(seq_fts, torch.empty(1, out_sz, device=device).normal_())  # [N,1]
    f_2 = F.linear(seq_fts, torch.empty(1, out_sz, device=device).normal_())  # [N,1]

    logits = f_1 + f_2.T

    mask = (adj_mat > 0).float()
    masked_logits = F.leaky_relu(logits) + (-1e9) * (1.0 - mask)
    coefs = F.softmax(masked_logits, dim=-1)

    if coef_drop != 0.0:
        coefs = F.dropout(coefs, p=coef_drop, training=True)

    ret = torch.matmul(coefs, seq_fts)
    bias = torch.zeros(out_sz, device=device)
    ret = ret + bias

    if residual:
        if seq.shape[1] == out_sz:
            ret += seq
        else:
            res = F.linear(seq, torch.empty(out_sz, D, device=device).normal_(), bias=None)
            ret += res

    return activation(ret)


class GCN(nn.Module):
    def __init__(
            self,
            struct,
            dropout_rate=0.5,
            alpha=100.0,
            beta=1.0,
            gamma=10.0,
            eta=25.0,
            learning_rate=0.001,
    ):
        super(GCN, self).__init__()
        self.struct = struct
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.learning_rate = learning_rate

        self.best_mes = 0.0

        self.layers = len(self.struct)
        self.W = nn.ParameterDict()
        self.b = nn.ParameterDict()

        # Encoder
        self. gc_encoder_layers = nn.ModuleList()
        for i in range(self.layers - 1):
            self.gc_encoder_layers.append(GraphConv(self.struct[i], self.struct[i+1]))


        # Decoder (reverse)
        self.struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = nn.Parameter(torch.randn([struct[i], struct[i + 1]]))
            self.b[name] = nn.Parameter(torch.zeros([struct[i + 1]]))
        self.struct.reverse()

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def forward(self, X, adj, current_mes=None):
        H = self.encode(X, adj)
        X_recon = self.decode(H)

        # Loss components
        loss_emb = self.get_emb_loss(H, adj)
        loss_res = self.get_res_loss(X, X_recon)
        loss_gcn = self.alpha * loss_res + self.beta * loss_emb

        if current_mes is not None:
            self.best_mes = max(self.best_mes, current_mes)
            loss_mes = self.get_mes_loss(loss_gcn)
            loss = loss_gcn + self.gamma * loss_mes
        else:
            loss = loss_gcn

        return loss, H, X_recon

    def encode(self, X, adj):
        for i, layer in enumerate(self.gc_encoder_layers):
            X = layer(X, adj)
            X = F.leaky_relu(X)
            if i == len(self.gc_encoder_layers) - 2:
                X = attn_head(X, self.struct[i + 1], adj, activation=F.leaky_relu)

            if i < len(self.gc_encoder_layers) - 1:
                X = F.dropout(X, p=self.dropout_rate, training=self.training)
        return X

    def decode(self, H):
        X = H
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            X = torch.sigmoid(torch.matmul(X, self.W[name]) + self.b[name])
        return X

    def get_emb_loss(self, H, adj):
        D = torch.diag(adj.sum(1))
        L = D - adj
        return 2 * torch.trace(H.t() @ L @ H)

    def get_res_loss(self, X, X_recon):
        B = X * (self.eta - 1) + 1
        return torch.sum(((X_recon - X) * B) ** 2)

    def get_mes_loss(self, loss_res):
        loss_mag = torch.log10(loss_res) - 3
        scale = 10 ** torch.ceil(loss_mag)
        return (1 - self.best_mes) * scale

def fit(model, data, mes=None, device='cpu'):
    x = torch.tensor(data['X'], dtype=torch.float32, device=device)
    adj = torch.tensor(data['adj'], dtype=torch.float32, device=device)


    model.train()
    model.optimizer.zero_grad()

    loss, _, _ = model(x, adj, current_mes=mes)

    loss.backward()
    model.optimizer.step()

    return loss.item()

def get_loss(model, data, device='cpu'):
    x = torch.tensor(data['X'], dtype=torch.float32, device=device)
    adj = torch.tensor(data['adj'], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        loss, _, _ = model(x, adj)
    return loss.item() * 0.000000001

def get_embedding(model, data, device='cpu'):
    x = torch.tensor(data['X'], dtype=torch.float32, device=device)
    adj = torch.tensor(data['adj'], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        emb = model.encode(x, adj)  # self.H
    return emb.cpu().numpy()

