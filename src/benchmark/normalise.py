# Use this script in order to generate normalisation .json files to use with the dataloader

# First, set up the dataloader as you would for your application.

########################################################################################################################
# User-modification START
########################################################################################################################

import numpy as np
import dill
import datetime
import multiprocessing
import os

from src.dataloader import Dataset

n_procs = 4 # Set to number of available CPUs
expname = "sample_datasets"

datapath = ["PATH TO ERA5625 SAMPLES DILL FILE",
           "PATH TO IMERG5625 SAMPLES DILL FILE",
            "PATH TO SIMSAT5625 SAMPLES DILL FILE"]

# partition_conf = {"train":
#                         {"timerange": (datetime.datetime(2010, 1, 1, 0).timestamp(),
#                                        datetime.datetime(2010, 12, 31, 0).timestamp()),
#                          "increment_s": 60 * 60},
#                   "test":
#                         {"timerange": (datetime.datetime(2017, 1, 15, 0).timestamp(),
#                                        datetime.datetime(2018, 12, 31, 0).timestamp()),
#                          "increment_s": 60 * 60}}
#partition_type = "range"

partition_conf = {"timerange": (datetime.datetime(2018, 1, 1, 0).timestamp(),
                                datetime.datetime(2019, 12, 31, 23).timestamp()),
                  # Define partition elements
                  "partitions": [{"name": "train", "len_s": 12 * 24 * 60 * 60, "increment_s": 60 * 60},
                                 {"name": "val", "len_s": 2 * 24 * 60 * 60, "increment_s": 60 * 60},
                                 {"name": "test", "len_s": 2 * 24 * 60 * 60, "increment_s": 60 * 60}]}


partition_type = "repeat"

sample_conf = {"lead_time_{}".format(int(lt / 3600)):  # sample modes
    {
        "sample":  # sample sections
            {
                "lat2d": {"vbl": "era5625/lat2d"},
                "lon2d": {"vbl": "era5625/lon2d"},
                "orography": {"vbl": "era5625/orography"},
                "slt": {"vbl": "era5625/slt"},
                "lsm": {"vbl": "era5625/lsm"},  # sample variables
                # "lat": {"vbl": "era5625/lat2d"},
                "tp": {"vbl": "era5625/tp",
                       "t": np.array([lt]),
                       "interpolate": ["nan", "nearest_past", "nearest_future"][1],
                       "normalise": ["log"]},
                "imerg": {"vbl": "imerg5625/precipitationcal",
                          "t": np.array([lt]),
                          "interpolate": ["nan", "nearest_past", "nearest_future"][1],
                          "normalise": ["log"]},
                "clbt0": {"vbl": "simsat5625/clbt:0",
                          "t": np.array([lt]),
                          "interpolate": ["nan", "nearest_past", "nearest_future"][1],
                          "normalise": ["log"]},
                "clbt1": {"vbl": "simsat5625/clbt:1",
                          "t": np.array([lt]),
                          "interpolate": ["nan", "nearest_past", "nearest_future"][1],
                          "normalise": ["log"]},
                "clbt2": {"vbl": "simsat5625/clbt:2",
                          "t": np.array([lt]),
                          "interpolate": ["nan", "nearest_past", "nearest_future"][1],
                          "normalise": ["log"]},
            }
    }
    for lt in np.array([3, 7]) * 3600}  # np.array([1, 3, 6, 9]) * 3600}

# choose a default normalisation method
default_normalisation = "stdmean_global"

########################################################################################################################
# User-modification STOP
########################################################################################################################

if partition_type == "repeat":
    partition_labels = [v["name"] for v in partition_conf["partitions"]]
else:
    partition_labels = list(partition_conf.keys())

dataset = Dataset(datapath=datapath,
                  partition_conf=partition_conf,
                  partition_type=partition_type,
                  partition_selected="train",
                  sample_conf=sample_conf,
                  )
dataset_conf = dict(datapath=datapath,
                    partition_conf=partition_conf,
                    partition_type=partition_type,
                    partition_selected="train",
                    sample_conf=sample_conf)

# Go through all partitions and select all variables in use
vbls = {}
for i, partition in enumerate(partition_labels):
    vbls[partition] = set()
    print("Generating normalisation data for partition: {}  ({}/{})".format(partition, i, len(list(partition_conf.keys()))))
    dataset.select_partition(partition)
    for mode, mode_v in sample_conf.items():
        for section, section_v in mode_v.items():
            for k, v in section_v.items():
                for n in v.get("normalise", [default_normalisation]):
                    vbls[partition].add((v["vbl"], n, "t" in v))

# Retrieve the dataset idx for all all partitions
timesegments = {}
for i, partition in enumerate(partition_labels):
    timesegments[partition] = dataset.get_partition_ts_segments(partition)

# TODO: const normalisation!

# create a list of jobs to be done
joblist = []
for partition in partition_labels:
    for vbl in list(vbls[partition]):
        joblist.append({"timesegments": timesegments[partition],
                        "vbl_name": vbl[0],
                        "normalise": vbl[1],
                        "has_t": vbl[2],
                        "dataset_conf": dataset_conf,
                        "partition": partition})


def worker(args):
    # creating our own dataset per thread, alleviates any issues with memmaps and multiprocessing!
    dataset = Dataset(**args["dataset_conf"])
    dataset.select_partition(args["partition"])

    fi = None
    if args["has_t"]:
        # expand timesegments
        for ts in args["timesegments"]:
            ret = dataset.get_file_indices_from_ts_range(ts, args["vbl_name"])
            if fi is None:
                fi = ret
            else:
                fi = np.concatenate([fi, ret])
    else:
        fi = None

    vals = None
    if fi is not None:
        vals = dataset[args["vbl_name"]][fi]
    else:  # constant value
        vals = dataset[args["vbl_name"]]

    results = {args["vbl_name"]: {}}
    n = args["normalise"]
    if n in ["stdmean_global"]:
        mean = np.nanmean(vals) # will be done out-of-core automagically by numpy memmap
        std = np.nanstd(vals) # will be done out-of-core automagically by numpy memmap
        fn = lambda x: (x-mean) / std if std != 0.0 else (x-mean)
        results[args["vbl_name"]]["stdmean_global"] = {"mean": mean, "std": std, "fn": fn}
    elif n in ["log"]:
        std = np.nanstd(vals) # will be done out-of-core automagically by numpy memmap
        fn = lambda x: np.log(max(x, 0.0) / std + 1)
        results[args["vbl_name"]]["log"] = {"std": std, "fn": fn}
    else:
        print("Unknown normalisation: {}".format(n))

    return dill.dumps({args["partition"]: results})


pool = multiprocessing.Pool(processes=n_procs)
results = pool.map(worker, joblist)

results_dct = {}
for r in results:
    loadr = dill.loads(r)
    partition = list(loadr.keys())[0]
    if partition not in results_dct:
        results_dct[partition] = loadr[partition]
    else:
        results_dct[partition].update(loadr[partition])

# save to normalisation file
with open(os.path.join("normalisations", "normalisations_{}.dill".format(expname)), "wb") as f:
    dill.dump(results_dct, f)
