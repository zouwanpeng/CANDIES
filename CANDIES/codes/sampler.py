import numpy as np
import torch
from tqdm import tqdm


def model_sample_diff(model, device, dataloader, total_sample, time, is_condi, condi_flag, mask_time=None):
    noise = []
    i = 0
    for _, x_hat, x_cond in dataloader:  # Calculate the noise of the entire shape and calculate the batch size in one cycle
        x_hat, x_cond = x_hat.float().to(device), x_cond.float().to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)

        if not is_condi:
            n = model(total_sample[i:i + len(x_cond)], t, None)  # Noise of batch size
        else:
            n = model(total_sample[i:i + len(x_cond)], x_hat, t, x_cond, condi_flag=condi_flag)
        noise.append(n)
        i = i + len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise


def sample_diff(model,
                dataloader,
                noise_scheduler,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=True,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                bias=None,
 ):
    model.eval()
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]  # Reverse

    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            # output noise
            model_output = model_sample_diff(model,
                                             device=device,
                                             dataloader=dataloader,
                                             total_sample=x_t,  # x_t
                                             time=time,  # t
                                             is_condi=is_condi,
                                             condi_flag=True)

        # calculate x_{t-1}
        x_t, _ = noise_scheduler.step(model_output / bias,  # noise often，bias is 'exposure bias'
                                      torch.from_numpy(np.array(time)).long().to(device),
                                      x_t,
                                      model_pred_type=model_pred_type)

    recon_x = x_t.detach().cpu().numpy()
    return recon_x
