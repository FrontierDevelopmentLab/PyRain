# RainBench
You should update the data paths in config.yml.

## Forecasting Precipitation from ERA
Specify `source` as {'simsat', 'simsat_era', 'era16_3'} to use data (*from 2016*) from Simsat alone, ERA5 alone, or both Simsat and ERA5, respectively. 

To use all data available in ERA5 for training (*from 1971*), set `source` as 'era'.

Set `inc_time` to concatenate inputs with hour, day, month.

For example, to train, run

```
python3 run_benchmark.py --sources simsat_era --inc_time --config_file config.yml
```

## Forecasting Precipitation from IMERG
Again, specify `source` as {'simsat', 'simsat_era', 'era16_3'} to use data (*from 2016*) from Simsat alone, ERA5 alone, or both Simsat and ERA5, respectively. 

To use all data available in ERA5 for training (*from 2000*), set `source` as 'era'.

For predicting IMERG precipitation, we found empirically that removing the relu function at the end of the ConvLSTM works better.

Set `inc_time` to concatenate inputs with hour, day, month.

```
python3 run_benchmark.py --sources simsat_era --no_relu --imerg --inc_time --config_file config.yml
```

## Evaluating trained models

To evaluate trained models on the test set, run the following.

```
python3 run_benchmark.py --test --phase test --load {MODEL_PATH}
```
