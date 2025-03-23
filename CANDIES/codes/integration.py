import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm


class Encode_all(Module):

    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2,
                 dropout=0.0, act=F.relu):
        super(Encode_all, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.atten_inte = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2,
                adj_feature_omics2):
        # Encode spatial graph
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)

        # Encode feature graph
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # Combine embeddings
        emb_latent_spatial_combined = torch.mean(torch.stack([emb_latent_spatial_omics1, emb_latent_spatial_omics2]), dim=0)
        emb_latent_feat_combined = torch.mean(torch.stack([emb_latent_feature_omics1, emb_latent_feature_omics2]), dim=0)

        # Between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_inte(emb_latent_spatial_combined, emb_latent_feat_combined)

        # Decode to reconstruct original features
        emb_recon_spatial_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_spatial_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        emb_recon_feature_omics1 = self.decoder_omics1(emb_latent_combined, adj_feature_omics1)
        emb_recon_feature_omics2 = self.decoder_omics2(emb_latent_combined, adj_feature_omics2)

        # Return all embeddings and reconstructed features
        results = {
            'emb_latent_spatial_omics1': emb_latent_spatial_omics1,
            'emb_latent_spatial_omics2': emb_latent_spatial_omics2,
            'emb_latent_feature_omics1': emb_latent_feature_omics1,
            'emb_latent_feature_omics2': emb_latent_feature_omics2,
            'emb_latent_spatial_combined': emb_latent_spatial_combined,
            'emb_latent_feat_combined': emb_latent_feat_combined,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_spatial_omics1': emb_recon_spatial_omics1,
            'emb_recon_spatial_omics2': emb_recon_spatial_omics2,
            'emb_recon_feature_omics1': emb_recon_feature_omics1,
            'emb_recon_feature_omics2': emb_recon_feature_omics2,
            'alpha_omics_1_2': alpha_omics_1_2
        }

        return results

    class NTXentLoss(nn.Module):
        """
        NT-Xent loss for self-supervised learning in SimCLR.

        Parameters
        ----------
        temperature
            For the softmax in InfoNCE loss.
        mask_fill
            The value to fill the mask with.

        References
        ----------
        - https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """

        def __init__(self, temperature: float = 0.07):
            super().__init__()
            self.temperature = temperature
            self.n_modality = 2
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
            r"""
            Compute the contrastive loss.

            Parameters
            ----------
            z1
                The output of the first modality. (N, D)
            z2
                The output of the second modality. (N, D)

            Returns
            ----------
            The contrastive loss.
            """
            batch_size = z1.shape[0]
            labels = torch.cat([torch.arange(batch_size)] * self.n_modality, dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(z1.device)  # (2N, 2N)

            features = torch.cat([z1, z2], dim=0)
            features = F.normalize(features, dim=1)

            similarity_matrix = torch.matmul(features, features.T)  # (2N, 2N)
            assert similarity_matrix.shape == (
                self.n_modality * batch_size, self.n_modality * batch_size)
            assert similarity_matrix.shape == labels.shape

            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)  # (2N, 2N)
            labels = labels[~mask].view(labels.shape[0], -1)  # (2N, 2N - 1), rm diagonal
            similarity_matrix = similarity_matrix[~mask].view(
                similarity_matrix.shape[0], -1
            )  # (2N, 2N - 1), remove the diagonal
            assert similarity_matrix.shape == labels.shape

            # select and combine multiple positives, (2N, 1)
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

            # select only the negatives, (2N, 2N - 2), remove positive pairs
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)

            logits = logits / self.temperature

            return self.criterion(logits, labels)


class Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class Decoder(Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class AttentionLayer(Module):
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.

    Returns
    -------
    Aggregated representations and modality weights.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


def train_and_infer(features_omics1, features_omics2,
                    adj_spatial_omics1, adj_feature_omics1,
                    adj_spatial_omics2, adj_feature_omics2,
                    epochs, device,
                    dim_out_feat_omics1=64, dim_out_feat_omics2=64,
                    learning_rate=0.001, weight_decay=0.00,
                    patience=10, min_delta=0.001
                    ):
    """
    Train the model using embeddings to calculate losses during training and return the final combined embeddings.

    Args:
        features_omics1: Input features for omics1.
        features_omics2: Input features for omics2.
        adj_spatial_omics1: Spatial adjacency matrix for omics1.
        adj_feature_omics1: Feature adjacency matrix for omics1.
        adj_spatial_omics2: Spatial adjacency matrix for omics2.
        adj_feature_omics2: Feature adjacency matrix for omics2.
        epochs: Number of training epochs.
        device: Device to run training and inference (e.g., 'cuda' or 'cpu').
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in loss to be considered an improvement.

    Returns:
        Final combined embeddings and training losses.
    """

    # Initialize model
    model = Encode_all(
        dim_in_feat_omics1=features_omics1.shape[1],
        dim_out_feat_omics1=dim_out_feat_omics1,
        dim_in_feat_omics2=features_omics2.shape[1],
        dim_out_feat_omics2=dim_out_feat_omics2
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate,
                                 weight_decay=weight_decay)

    # Move data and model to device
    model = model.to(device)
    features_omics1 = features_omics1.to(device)
    features_omics2 = features_omics2.to(device)
    adj_spatial_omics1 = adj_spatial_omics1.to(device)
    adj_feature_omics1 = adj_feature_omics1.to(device)
    adj_spatial_omics2 = adj_spatial_omics2.to(device)
    adj_feature_omics2 = adj_feature_omics2.to(device)

    contrastive_loss = model.NTXentLoss()

    best_loss = float('inf')
    patience_counter = 0

    # Training loop
    model.train()
    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            model.train()
            # Forward pass
            results = model(features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                            adj_spatial_omics2, adj_feature_omics2)

            # Extract embeddings
            emb_latent_spatial_omics1 = results['emb_latent_spatial_omics1']
            emb_latent_spatial_omics2 = results['emb_latent_spatial_omics2']
            emb_latent_feature_omics1 = results['emb_latent_feature_omics1']
            emb_latent_feature_omics2 = results['emb_latent_feature_omics2']

            # Extract reconstructed embeddings
            emb_recon_spatial_omics1 = results['emb_recon_spatial_omics1']
            emb_recon_spatial_omics2 = results['emb_recon_spatial_omics2']
            emb_recon_feature_omics1 = results['emb_recon_feature_omics1']
            emb_recon_feature_omics2 = results['emb_recon_feature_omics2']

            # Reconstruction losses
            recon_loss_spatial_omics1 = F.mse_loss(emb_recon_spatial_omics1, features_omics1)
            recon_loss_spatial_omics2 = F.mse_loss(emb_recon_spatial_omics2, features_omics2)
            recon_loss_feature_omics1 = F.mse_loss(emb_recon_feature_omics1, features_omics1)
            recon_loss_feature_omics2 = F.mse_loss(emb_recon_feature_omics2, features_omics2)

            # Contrastive losses
            contrastive_loss_spatial = contrastive_loss(emb_latent_spatial_omics1, emb_latent_spatial_omics2)
            contrastive_loss_feature = contrastive_loss(emb_latent_feature_omics1, emb_latent_feature_omics2)

            # Total loss
            total_recon_loss = recon_loss_spatial_omics1 + recon_loss_spatial_omics2 + recon_loss_feature_omics1 + recon_loss_feature_omics2
            total_contra_loss = contrastive_loss_spatial + contrastive_loss_feature
            total_loss = total_recon_loss + total_contra_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Check for early stopping
            if total_loss.item() + min_delta < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered at epoch:", epoch + 1)
                    break

            # Update tqdm with all losses
            pbar.set_postfix({
                "Total Loss": total_loss.item(),
                "Contrastive Loss (Spatial)": contrastive_loss_spatial.item(),
                "Contrastive Loss (Feature)": contrastive_loss_feature.item(),
                "Reconstruction Loss 1": recon_loss_spatial_omics1.item(),
                "Reconstruction Loss 2": recon_loss_spatial_omics2.item(),
                "Reconstruction Loss 3": recon_loss_feature_omics1.item(),
                "Reconstruction Loss 4": recon_loss_feature_omics2.item(),
            })
            pbar.update(1)

    # Inference step
    model.eval()
    with torch.no_grad():
        final_results = model(features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                              adj_spatial_omics2, adj_feature_omics2)

    return final_results
