import torch
from torchvision.utils import make_grid

def plot_random_outputs_multi_ts(sample_X, sample_y, pred_y,
        idx_dictionary, normalizer, order):
    """
    X of shape [N, seq_len, channels, lat, lon]
    y of shape [N, channels, lat, lon]
    """
    num_lead_times = len(sample_X)
    sample_images = []
    for v in order:
        _, cat_ind_y = idx_dictionary[v]
        truth_v = sample_y[:, cat_ind_y]
        pred_v = pred_y[:, cat_ind_y]
        diff_v = (truth_v - pred_v).abs()
        
        # scale for the image
        vmin = min([pred_v.min(), truth_v.min()])
        vmax = max([pred_v.max(), truth_v.max()])
        scale = lambda x: (x - vmin) / (vmax - vmin)

        # truth
        # sample_images += [scale(sample_X_v[:, ts]) for ts in range(seq_len)]
        sample_images += [scale(torch.unsqueeze(truth_v[i], 0)) for i in range(num_lead_times)]
        sample_images += [scale(torch.unsqueeze(pred_v[i], 0)) for i in range(num_lead_times)]
        sample_images += [scale(torch.unsqueeze(diff_v[i], 0)) for i in range(num_lead_times)]

    nrow = num_lead_times
    grid = make_grid(sample_images, nrow=nrow)
    return grid

