# Update: Data now publicly available!

We are very happy to announce that the memmap datasets are now available publicly at:
https://console.cloud.google.com/storage/browser/aaai_release
You require an ordinary Google account to access them.

The data comes in two different resolutions, `5.625` degrees, and `1.40625` degrees.
To see what variables (and timeranges) are contained in each dataset, simply download the associated .dill file and read out as follows (python3):

```import dill
import pprint
with open("path-to-dill-file", "rb") as f:
    info = dill.load(f)
pprint.pprint(info)```

Please let us know if you have any questions/issues - for technical issues please use the github issues.
Many thanks, and we hope you will find RainBench useful!

# RainBench - Getting Started

## Downloading the Dataset
Please register [here](https://forms.gle/3AdMJsKtuJ8M1E1Y8) to download the RainBench dataset.

After downloading, you should update the data paths in config.yml.

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


# Visualizing Predictions

To visualize the predictions, run the following. 

```
python3 -m src.benchmark.plot_outputs --load {MODEL_PATH} --nc_file {ANY_NC_FILE_PATH}
```

Example predictions for a randome test date (12 July 2019) is shown below:

### Truth
![](https://i.imgur.com/O1Fk0XS.gif)

### Simsat
![](https://i.imgur.com/uMvodFI.gif)

### ERA
![](https://i.imgur.com/UbOe0Ia.gif)

### Simsat & ERA
![](https://i.imgur.com/tX5pmLP.gif)

# Advanced Topics

## Going to higher spatial resolution

RainBench contains memmap datasets at two different spatial resolutions: 5.625 degrees, and 1.46025 degrees. 
Fortunately, the NetCDF-->Memmap conversion scripts for 5.625 degrees that come with RainBench can be easily adjusted to NetCDF datasets at higher - or native - resolution. The main change needing to be done is to adjust the pixel width and height of the different variable channels. As the conversion scripts use use multiprocessing in order to saturate I/O during dataset conversion, even very high resolution datasets can be fine-grainedly converted to Memmaps.

## Generating normalisation files
Under `src/benchmark/normalise.py`, you can generate your own normalisation files to be used for on-the-fly normalisation of training data. Simply insert your own sample configuration and partitioning setup into the section marked and run the file using python3. This will generate a pickled `dill` file, which contains a dictionary with normalisation entries (and indeed, packaged functions) for each variable field across each partition. Partition of type `repeat` are expressly supported. Just as data conversion, normalisation supports multiprocessing (and out-of-core computations), meaning even datasets at large resolutions can be handled. It is also easy to add new normalisation routines in the fields provided (also have a look at `src/benchmark/transforms.py` for patch-wise local normalisation techniques).

