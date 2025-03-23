import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm

from .preprocess1 import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, \
    construct_interaction_KNN, get_feature

from .DiTs import seed_everything


class ZINBLoss(nn.Module):

    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, device=None):
        eps = 1e-10
        scale_factor = torch.Tensor([1.0]).to(device)
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)



class EnDecoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(EnDecoder, self).__init__()
        self.in_features = in_features
        # self.hidden_features = hidden_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        # encoder weights
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        # decoder
        self._dec_mean = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            MeanAct()
        )
        self._dec_disp = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            DispAct()
        )
        self._dec_pi = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def encode(self, feat, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        hiden_emb = z
        return hiden_emb

    def decode(self, hiden_emb, adj):
        h = torch.mm(hiden_emb, self.weight2)
        h = torch.mm(adj, h)

        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        return h, _mean, _disp, _pi


class encoder_ZINB():
    def __init__(self,
                 adata,
                 device='cuda:0',
                 learning_rate=0.0005,
                 weight_decay=0.00,
                 n_top_genes=3000,
                 epochs=500,
                 random_seed=41,
                 alpha=10,
                 datatype='10X',
                 dim_output=None,
                 ):

        self.adata = adata.copy()
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.alpha = alpha
        self.n_top_genes = n_top_genes
        self.datatype = datatype

        self.zinb_loss = ZINBLoss().cuda()

        seed_everything(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata, self.n_top_genes)
        #
        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)
        #
        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)

    def train(self):
        # initialize Encoder 和 Decoder
        self.model = EnDecoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()
        copyemb = None

        for epoch in tqdm(range(self.epochs)):
            self.hiden_feat = self.model.encode(self.features, self.adj)

            self.x_recon, meanbatch, dispbatch, pibatch = self.model.decode(self.hiden_feat, self.adj)

            if epoch == 0:
                copyemb = self.hiden_feat

            # autoencoder loss
            self.re_loss = F.mse_loss(self.features, self.x_recon)
            zinb_loss = self.zinb_loss(self.features, meanbatch, dispbatch, pibatch, device=self.device)
            loss = (self.alpha * self.re_loss + 0.5 * zinb_loss) * 0.2
            nan_count = torch.isnan(self.x_recon).sum()

            if nan_count.item() > 0:
                self.hiden_feat = copyemb
                print('Early stop!')
                break

            copyemb = self.hiden_feat

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished")

        with torch.no_grad():
            self.model.eval()
            return self.hiden_feat.detach().cpu().numpy(), self.adj.detach().cpu().numpy()
