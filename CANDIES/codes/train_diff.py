import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm

from .diff_scheduler import NoiseScheduler
from .DiTs import DiT_diff
from .sampler import sample_diff

class ConditionalDiffusionDataset():
    def __init__(self, adata_omics1, adata_omics2):
        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2

        self.st_sample = torch.tensor(self.adata_omics1, dtype=torch.float32)
        self.con_sample = torch.tensor(self.adata_omics2, dtype=torch.float32)
        self.con_data = torch.tensor(self.adata_omics2, dtype=torch.float32)

    def __len__(self):
        return len(self.adata_omics1)

    def __getitem__(self, idx):
        return self.st_sample[idx], self.con_sample[idx], self.con_data

class diffusion_loss(nn.Module):
    def __init__(self):
        super(diffusion_loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss_mse = self.mse(y_pred, y_true)
        return loss_mse


def normal_train_diff(model,
                      dataloader,
                      lr: float = 1e-4,
                      num_epoch: int = 1400,
                      pred_type: str = 'noise',
                      diffusion_step: int = 1000,
                      device=torch.device('cuda:0'),
                      is_tqdm: bool = True,
                      patience: int = 20):
    """

    Args:
        lr (float):
        momentum (float): momentum
        max_iteration (int, optional): iteration of training. Defaults to 30000.
        pred_type (str, optional): prediction type noise or x_0. Defaults to 'noise'.
        batch_size (int, optional): Defaults to 1024.
        diffusion_step (int, optional): diffusion step number. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): whether to use condition. Defaults to False.
        is_tqdm (bool, optional): enable progress bar. Defaults to True.
        is_tune (bool, optional): whether to use ray tune. Defaults to False.
        condi_drop_rate (float, optional): whether to use classifier free guidance to set drop rate. Defaults to 0..
        patience (int): The number of epochs to tolerate early stopping. If the validation loss does not improve for 'patience' epochs, stop training.

    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = diffusion_loss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    best_loss = float('inf')
    patience_counter = 0

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.0
        for i, (x, x_hat, x_cond) in enumerate(dataloader):
            x, x_hat, x_cond = x.float().to(device), x_hat.float().to(device), x_cond.float().to(device)

            x_noise = torch.randn(x.shape).to(device)

            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).to(device)

            x_t = noise_scheduler.add_noise(x, x_noise, timesteps=timesteps)
            x_noisy = x_t

            noise_pred = model(x_noisy, x_hat, t=timesteps.to(device), y=x_cond)

            loss = criterion(x_noise, noise_pred)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)
        current_lr = optimizer.param_groups[0]['lr']

        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.7f}, lr:{current_lr:.2e}')

        # early stop
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop(patience:{patience})!!")
                break


def run_diff(dataset, k=3, batch_size=32, learning_rate=1e-4, num_epoch=1000,
             diffusion_step=500, hidden_size=256, depth=16, head=8, pca_dim=50,
             device='cuda:0', classes=6, patience=20, bias=None):
    """
    Run k-fold for training the DiT model.

    Parameters:
    - dataset: The dataset to train and test on (assumed to be an instance of ConditionalDiffusionDataset).
    - k: Number of folds.
    - batch_size: Batch size for training and testing.
    - learning_rate: Learning rate for the optimizer.
    - num_epoch: Number of epochs for training.
    - diffusion_step: Number of diffusion steps for the model.
    - hidden_size: Hidden size for the model.
    - depth: Number of layers in the model.
    - head: Number of attention heads.
    - pca_dim: Number of dimensions for PCA (if used).
    - mask_nonzero_ratio: Ratio for masking non-zero values.
    - mask_zero_ratio: Ratio for masking zero values.
    - device: The device to run the model on ('cuda' or 'cpu').
    - classes: Number of output classes for classification.

    Returns:
    - res_mtx: A matrix to store the results across all folds.
    """

    res_mtx = np.zeros((len(dataset), dataset.adata_omics1.shape[1]))  # Initialize result matrix
    splits = KFold(n_splits=k, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print(f'*******************  Fold {fold + 1}  (total {k} folds)  ********************')

        # Create data samplers for training and validation sets
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = val_idx

        # DataLoader for training and validation
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        # Model initialization
        input_size = dataset.adata_omics1.shape[1]
        condi_input_size = dataset.adata_omics2.shape[1]

        model = DiT_diff(
            st_input_size=input_size,
            condi_input_size=condi_input_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=head,
            mlp_ratio=4.0,
            classes=6,
            dit_type='dit'
        )

        model.to(device)
        model.train()

        # Train the model
        normal_train_diff(
            model,
            dataloader=train_dataloader,
            lr=learning_rate,
            num_epoch=num_epoch,
            diffusion_step=diffusion_step,
            device=device,
            pred_type='noise',
            patience=patience
        )

        # Initialize noise scheduler for sampling
        noise_scheduler = NoiseScheduler(num_timesteps=diffusion_step, beta_schedule='cosine')

        model.eval()

        # Collect test
        all_data = []
        for batch in test_dataloader:
            data, _, _ = batch
            all_data.append(data)
        test_gt = torch.cat(all_data, dim=0)

        # Sample the model and get predictions
        prediction = sample_diff(
            model,
            device=device,
            dataloader=test_dataloader,
            noise_scheduler=noise_scheduler,
            num_step=diffusion_step,
            sample_shape=(test_gt.shape[0], test_gt.shape[1]),
            is_condi=True,
            sample_intermediate=diffusion_step,
            model_pred_type='noise',
            bias=bias,
        )

        # Store the results in the matrix
        res_mtx[val_idx, :] = prediction

    return res_mtx
