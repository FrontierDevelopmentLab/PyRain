"""
Train model for benchmark tasks.
"""
from argparse import ArgumentParser, FileType
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import json
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers
from src.benchmark.utils import add_device_hparams, get_lat2d, add_yml_params
from src.benchmark.collect_data import get_data, get_checkpoint_path
from src.benchmark.models import ConvLSTMForecaster
from src.benchmark.graphics import plot_random_outputs_multi_ts
from src.benchmark.metrics import eval_loss, define_loss_fn, collect_outputs

class RegressionModel(pl.LightningModule):
    """
    Regression Module
    """
    def __init__(self, hparams, train_set, valid_set, normalizer, collate, lat2d=None):
        super().__init__()
        hparams['relu'] = not hparams['no_relu']
        self.hparams = hparams
        self.lead_times = hparams['lead_times']
        self.normalizer = normalizer
        self.categories = hparams['categories']
        self.trainset = train_set
        self.validset = valid_set
        self.normalizer = normalizer
        self.collate = collate
        self.multi_gpu = hparams['multi_gpu']
        self.target_v = self.categories['output'][0]
        
        self.net = ConvLSTMForecaster(
                        in_channels=hparams['num_channels'],
                        output_shape=(hparams['out_channels'], *hparams['latlon']),
                        channels=(hparams['hidden_1'], hparams['hidden_2']),
                        last_ts=True,
                        last_relu=hparams['relu'])
        
        self.plot = self.hparams['plot']
        if self.plot:
            # define dictionary to hold column names in input and output: {var_name: (input_col_index, output_col_index)}
            self.idxs = {}
            for ind_y, v in enumerate(self.categories['output']):
                self.idxs[v] = (self.categories['input'].index(v), ind_y) if v in self.categories['input'] else (None, ind_y)
            for ind_x, v in enumerate(self.categories['input']):
                if v not in self.categories['output']:
                    self.idxs[v] = (ind_x, None)
        
        if lat2d is None:
            lat2d = get_lat2d(hparams['grid'], self.validset.dataset)
        self.weights_lat, self.loss = define_loss_fn(lat2d)
        self.lat2d = lat2d
    
    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_nb):
        inputs, output, lts = batch
        pred = self(inputs.contiguous())
        results = eval_loss(pred, output, lts, self.loss, self.lead_times, phase='train', target_v=self.target_v, normalizer=self.normalizer)
        return {'loss': results['train_loss'], 'log': results, 'progress_bar': results}

    def validation_step(self, batch, batch_idx):
        inputs, output, lts = batch
        pred = self(inputs)
        results = eval_loss(pred, output, lts, self.loss, self.lead_times, phase='val', target_v=self.target_v, normalizer=self.normalizer)
        return results

    def test_step(self, batch, batch_idx):
        inputs, output, lts = batch
        pred = self(inputs)
        results = eval_loss(pred, output, lts, self.loss, self.lead_times, phase='test', target_v=self.target_v, normalizer=self.normalizer)
        return results

    def plot_outputs_on_tensorboard(self):
        samples = []
        for lt in self.hparams['lead_times']:
            sample_lt = self.validset.get_sample_at(f'lead_time_{lt}', datetime(2018, 7, 12, 0).timestamp())
            sample_lt['__sample_modes__'] = f'lead_time_{lt}'
            samples.append([sample_lt])
        sample = self.collate(samples)
        sample_X, sample_y, _ = sample
        pred_y = self(sample_X.cuda()).cpu()
        grid = plot_random_outputs_multi_ts(sample_X, sample_y, pred_y, self.idxs, self.normalizer, self.categories['output'])
        self.logger.experiment.add_image('generated_images', grid, self.global_step)
        
    def validation_epoch_end(self, outputs):
        log_dict = collect_outputs(outputs, self.multi_gpu)
        results = {'log': log_dict, 'progress_bar': {'val_loss': log_dict['val_loss']}}
        results = {**results, **log_dict}

        if self.plot:
            self.plot_outputs_on_tensorboard()
        return results

    def test_epoch_end(self, outputs):
        log_dict = collect_outputs(outputs, self.multi_gpu)
        results = {'log': log_dict, 'progress_bar': {'test_loss': log_dict['test_loss']}}
        results = {**results, **log_dict}
        return results
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams['lr'])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=False)

    @staticmethod
    def load_model(log_dir, **params):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param params: list of named arguments, used to update the model hyperparameters
        """
        # load hparams
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            hparams = yaml.load(fp, Loader=yaml.Loader)
            hparams.update(params)

        # load data
        hparams, loaderDict, normalizer, collate = get_data(hparams, tvt='train_valid_test')
        
        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        print(f'Loading model {model_path.parent.stem}')
        train_set = loaderDict['train']
        valid_set = loaderDict['valid']
        model = RegressionModel.load_from_checkpoint(str(model_path), hparams=hparams, \
            train_set=train_set, valid_set=valid_set, normalizer=normalizer, collate=collate)
        return model, hparams, loaderDict, normalizer, collate


def main(hparams):
    hparams = vars(hparams)
    hparams, loaderDict, normalizer, collate = get_data(hparams)
    
    # ------------------------
    # Model
    # ------------------------
    add_device_hparams(hparams)

    # define logger
    Path(hparams['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(hparams['log_path'], version=hparams['version'])
    logger.log_hyperparams(params=hparams)

    # define model
    model = RegressionModel(hparams, loaderDict['train'], loaderDict['valid'], normalizer, collate)

    chkpt = None if hparams['load'] is None else get_checkpoint_path(hparams['load'])
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=logger,
        max_epochs=hparams['epochs'],
        distributed_backend=hparams['distributed_backend'],
        precision=16 if hparams['use_amp'] else 32,
        default_root_dir=hparams['log_path'],
        deterministic=True,
        resume_from_checkpoint=chkpt,
        auto_lr_find=hparams['auto_lr'],
        auto_scale_batch_size=hparams['auto_bsz']
    )
    trainer.fit(model)


def main_test(hparams):
    assert (hparams.load is not None) and (hparams.phase is not None)
    phase = hparams.phase
    log_dir = hparams.load
    hparams = vars(hparams)
    add_device_hparams(hparams)

    # Load trained model
    print(f'Loading from {log_dir} to evaluate {phase} data.')
    model, hparams, loaderDict, normalizer, collate = RegressionModel.load_model(log_dir, \
        multi_gpu=hparams['multi_gpu'], num_workers=hparams['num_workers'])
    trainer = pl.Trainer(
        gpus=hparams['gpus'],
        logger=None,
        max_epochs=hparams['epochs'],
        distributed_backend=hparams['distributed_backend'],
        default_root_dir=hparams['log_path'],
        deterministic=True
    )
    test_dataloader = DataLoader(loaderDict[phase], batch_size=hparams['batch_size'], \
        num_workers=hparams['num_workers'], collate_fn=collate, shuffle=False)

    # Evaluate the model
    test_results = trainer.test(model, test_dataloaders=test_dataloader)
    if isinstance(test_results, list):
        test_results = test_results[0]
    rmse = {'rmse_' + n: np.sqrt(test_results[n]) for n in test_results}
    test_results = {**rmse, **test_results}

    # Save evaluation results
    results_path = Path(log_dir) / f'{phase}_results.json'
    with open(results_path, 'w') as fp:
        json.dump(test_results, fp, sort_keys=True, indent=4)
    print('saved to ', results_path)


def main_baselines(hparams):
    """
    execute calculation for persistence / climatology baselines
    """
    assert hparams.phase is not None
    from src.benchmark.baseline_data import get_persistence_data, get_climatology_data
    phase = hparams.phase
    hparams = vars(hparams)
    add_device_hparams(hparams)

    if hparams['persistence']:
        loaderDict, dataloader, target_v, lead_times = get_persistence_data(hparams)
    else:
        same_pred, loaderDict, dataloader, target_v, lead_times = get_climatology_data(hparams)

    # define loss
    lat2d = get_lat2d(hparams['grid'], loaderDict[phase].dataset)
    loss = define_loss_fn(lat2d)
    
    # collect data and iterate through
    outputs = []
    if hparams['persistence']:
        for inputs, output, lts in tqdm(dataloader):
            results = eval_loss(inputs, output, lts, loss, lead_times)
            outputs.append(results)
    else:
        for inputs, output, lts in tqdm(dataloader):
            if len(inputs) < hparams['batch_size']:
                same_pred = same_pred[:len(inputs)]
            results = eval_loss(same_pred, output, lts, loss, lead_times)
            outputs.append(results)
    
    # collect results
    log_dict = collect_outputs(outputs, hparams['multi_gpu'])
    log_dict = {v: float(log_dict[v].detach().cpu()) for v in log_dict}
    print(log_dict)
    return log_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    # Data
    parser.add_argument("--persistence", action='store_true', help='Compute persistence baseline')
    parser.add_argument("--climatology", action='store_true', help='Compute climatology baseline')
    parser.add_argument("--sources", type=str, choices=['simsat_era', 'era16_3', 'simsat', 'era'], help='Input sources')
    parser.add_argument("--imerg", action='store_true', help='Predict precipitation from IMERG')
    parser.add_argument("--grid", type=float, default=5.625, choices=[5.625, 1.4], help='Data resolution')
    parser.add_argument("--sample_time_window", type=int, default=12, help="Duration of sample time window, in hours")
    parser.add_argument("--sample_freq", type=int, default=3, help="Data frequency within the sample time window, in hours")
    parser.add_argument("--forecast_time_window", type=int, default=120, help="Maximum lead time, in hours")
    parser.add_argument("--forecast_freq", type=int, default=24, help="Forecast frequency")
    parser.add_argument("--inc_time", action='store_true', help='Including hour/day/month in input')
    # 
    parser.add_argument('--config_file', default='./config.yml', type=FileType(mode='r'), help='Config file path')
    parser.add_argument('--data_paths', nargs='+', help='Paths for dill files')
    parser.add_argument('--norm_path', type=str, help='Path of json file storing  normalisation statistics')
    parser.add_argument('--log_path', type=str, help='Path of folder to log training and store model')

    # Model
    parser.add_argument("--hidden_1", type=int, default=384, help="No. of hidden units (lstm).")
    parser.add_argument("--hidden_2", type=int, default=32, help="No. of hidden units (fc).")
    parser.add_argument("--no_relu", action='store_true', help='Not using relu on last network layer')
    # Training
    parser.add_argument("--gpus", type=int, default=-1, help="Number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'), help='Backend for pytorch-lightning')
    parser.add_argument('--use_amp', action='store_true', help='If true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="No. of epochs to train")
    parser.add_argument("--num_workers", type=int, default=8, help="No. of dataloader workers")
    parser.add_argument("--test", action='store_true', help='Evaluate trained model')
    parser.add_argument("--load", type=str, help='Path of checkpoint directory to load')
    parser.add_argument("--phase", type=str, default='test', choices=['test', 'valid'], help='Which dataset to test on.')
    parser.add_argument("--auto_lr", action='store_true', help='Auto select learning rate.')
    parser.add_argument("--auto_bsz", action='store_true', help='Auto select batch size.')
    # Monitoring
    parser.add_argument("--version", type=str, help='Version tag for tensorboard')
    parser.add_argument("--plot", type=bool, help='Plot outputs on tensorboard')

    hparams = parser.parse_args()

    if hparams.config_file:
        add_yml_params(hparams)

    if hparams.test:
        main_test(hparams)
    elif hparams.persistence or hparams.climatology:
        main_baselines(hparams)
    else:
        main(hparams)

