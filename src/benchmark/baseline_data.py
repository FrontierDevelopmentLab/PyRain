import numpy as np
import torch
from torch.utils.data import DataLoader
from src.benchmark.utils import get_vbl_name
from src.benchmark.collect_data import define_data_paths, write_partition_conf, read_normalization_stats
from src.dataloader.memmap_dataloader import Dataset


def collate_fn_persistence(x_list, v):
    """
    return
        inputs = [bsz, channels, lat, lon]
        output = [bsz, channels, lat, lon]
    """
    categories={'input': [v], 'input_temporal': [v], 'input_static': [], 'output': [v]}
    output = []
    inputs = []
    lead_times = []
    
    for sample in x_list:
        output.append(np.concatenate([sample[0]['target'][v] for v in categories['output']], 1))
        inputs.append([sample[0]['label'][v] for v in categories['input']]) #
        lead_times.append(int(sample[0]['__sample_modes__'].split('_')[-1]))

        inputs[-1] = np.concatenate(inputs[-1], 1)

    inputs = torch.Tensor(np.concatenate(inputs))
    output = torch.Tensor(np.concatenate(output))
    lead_times = torch.Tensor(lead_times).long()
    return inputs, output, lead_times


def write_sample_conf_persistence(v: str,
            lead_times: list,
            interporlation: str = "nearest_past",
            grid: float = 5.625):
    """
    Write a sample configuration dictionary for calculating baselines.
    """
    sample_conf = {}
    samples = {var: \
               {"vbl": get_vbl_name(var, grid), \
                "t": np.array([0]), \
                "interpolate": interporlation} \
            for var in [v]}

    for lt in lead_times:
        sample_conf["lead_time_{}".format(int(lt/3600))] = {
            "label": samples,
            "target": {var: {"vbl": get_vbl_name(var, grid), "t": np.array([lt]), "interpolate": interporlation} \
                for var in [v]}
            }
    return sample_conf


def get_persistence_data(hparams):
    """Main function to get data for computing climatology baseline according to hparams"""
    # get data
    target_v = 'precipitationcal' if hparams['imerg'] else 'tp'
    phase = hparams['phase']
    datapath = hparams['data_paths']
    lead_times = np.arange(hparams['forecast_freq'], hparams['forecast_time_window'] + hparams['forecast_freq'], hparams['forecast_freq']) * 3600
    partition_conf = write_partition_conf(hparams['sources'], hparams['imerg'])
    sample_conf = write_sample_conf_persistence(target_v, lead_times)
    loaderDict = {p: Dataset(datapath=datapath,
                        partition_conf=partition_conf,
                        partition_type="range",
                        partition_selected=p,
                        sample_conf=sample_conf) for p in [phase]}
    # define collate and dataloader
    lead_times = lead_times //3600
    collate = lambda x: collate_fn_persistence(x, target_v)
    dataloader = DataLoader(loaderDict[phase], batch_size=hparams['batch_size'], \
        num_workers=hparams['num_workers'], collate_fn=collate, shuffle=False)
    return loaderDict, dataloader, target_v, lead_times



def get_climatology_data(hparams):
    """Main function to get data for computing climatology baseline according to hparams"""
    loaderDict, trd, target_v, lead_times = get_persistence_data(hparams)
    # get climatology value (mean over all trainin data)
    normalizer = read_normalization_stats(hparams['sources'], hparams['grid'], hparams['imerg'])
    mean_pred_v = normalizer[target_v]['mean']
    # get prediction matrix
    latlon = (32, 64) if hparams['grid'] == 5.625 else (128, 256)
    pred = torch.ones((hparams['batch_size'], 1, *latlon)) * mean_pred_v
    return pred, loaderDict, trd, target_v, lead_times
