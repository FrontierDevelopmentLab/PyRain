import numpy as np
import datetime

from src.dataloader import Dataset

datapath = ["PATH TO ERA5625 SAMPLES DILL FILE",
           "PATH TO IMERG5625 SAMPLES DILL FILE",
            "PATH TO SIMSAT5625 SAMPLES DILL FILE"]

partition_conf = {"train":
    {"timerange": (
        datetime.datetime(2010, 1, 1, 0).timestamp(), datetime.datetime(2010, 12, 31, 0).timestamp()),
        "increment_s": 60 * 60},
    "test":
        {"timerange": (datetime.datetime(2017, 1, 15, 0).timestamp(),
                       datetime.datetime(2018, 12, 31, 0).timestamp()),
         "increment_s": 60 * 60}}
partition_type = "range"

sample_conf = {"lead_time_{}".format(int(lt / 3600)):  # sample modes
    {
        "sample":  # sample sections
            {
                "lsm": {"vbl": "yera5625/lsm"},  # sample variables
                # "lat": {"vbl": "era5625/lat2d"},
                "t_300hPa": {"vbl": "yera5625/t:600hPa",
                             "t": np.array([0, -1, -2, -3, ]) * 3600,
                             "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                "t_500hPa": {"vbl": "yera5625/t",
                             "t": np.array([0, -1, -2, -3, ]) * 3600,
                             "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                # "t1000": {"vbl": "xera5625/t:1000hPa",
                #          "t": np.array([0, -1, -2, -3, -4]) * 3600,
                #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]}
            },
        "label":
            {
                "tp": {"vbl": "yera5625/tp",
                       "t": np.array([lt]),
                       "interpolate": ["nan", "nearest_past", "nearest_future"][1]}}
    }
    for lt in np.array([3, 7]) * 3600}  # np.array([1, 3, 6, 9]) * 3600}

# Met-Net style: different targets per label -- as an option

dataset = Dataset(datapath=datapath,
                  partition_conf=partition_conf,
                  partition_type=partition_type,
                  partition_selected="train",
                  sample_conf=sample_conf,
                  )

tp = dataset[((datetime.datetime(2018,1,1,0).timestamp(), datetime.datetime(2019,12,31,23).timestamp(), 3600), ["era5625/tp"], None)]
imerg = dataset[((datetime.datetime(2018,1,1,0).timestamp(), datetime.datetime(2019,12,31,23).timestamp(), 3600), ["imerg5625/precipitationcal"], None)]
simsat = dataset[((datetime.datetime(2018,1,1,0).timestamp(), datetime.datetime(2019,12,31,23).timestamp(), 3*3600), ["simsat5625/clbt:0"], {"interpolate":"nearest_past"})]
simsat2 = dataset[([datetime.datetime(2018,1,1,0).timestamp(), datetime.datetime(2019,12,31,23).timestamp()], ["simsat5625/clbt:0"], {})]
