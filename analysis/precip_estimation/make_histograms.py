from datetime import datetime
import numpy as np
import os, sys
import json
from scipy import stats
from multiprocessing import Pool, TimeoutError
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.fast_dataloader import Dataset


if __name__ == "__main__":

    # set up dataloader with any dataset type you can think of
    memmap_root = "" # SET MEMMAP DATA ROOT PATH HERE
    datapath = [os.path.join(memmap_root, "simsat5625", "simsat5625.dill"),
                os.path.join(memmap_root, "imerg5625", "imerg5625.dill"),
                os.path.join(memmap_root, "era5625", "era5625.dill"),
                ]

    daterange_train = (datetime(2016, 4, 1).timestamp(), datetime(2017, 12, 31, 23).timestamp())
    daterange_test = (datetime(2019, 1, 6, 0).timestamp(), datetime(2019, 12, 31, 23).timestamp())
    daterange_val = (datetime(2018, 1, 6, 0).timestamp(), datetime(2018, 12, 31, 23).timestamp())

    partition_conf = {"train":
                          {"timerange": daterange_train,
                           "increment_s": 60 * 60},
                      "val":
                          {"timerange": daterange_val,
                           "increment_s": 60 * 60},
                      "test":
                          {"timerange": daterange_test,
                           "increment_s": 60 * 60}}

    partition_type = "range"

    sample_conf = {"mode0":  # sample modes
        {
            "sample":  # sample sections
                {
                    "lsm": {"vbl": "era140625/lsm"},
                },
        }
    }

    dr = (datetime(2016, 4, 1).timestamp(), datetime(2019, 12, 31, 21).timestamp())

    part = "test"
    # read in every imerg frame and create a rain class histogram for each and save in a file in the end
    def get_histograms(args):
        dataset_indices, i = args
        print ("Starting process {} indices at iteration {}...".format(len(dataset_indices), i))

        dataset = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected=part,
                          sample_conf=sample_conf,
                          )

        res = []
        def seg_rain_imerg(frame):
            c0 = np.count_nonzero( (frame >= 0.0)  & (frame < 2.5))
            c1 =  np.count_nonzero((frame >= 2.5) & (frame < 10.0))
            c2 =  np.count_nonzero((frame >= 10.0) & (frame < 50.0))
            c3 =  np.count_nonzero((frame >= 50.0) & (frame < 500000.0))
            return c0, c1, c2, c3

        for data_idx in dataset_indices:
            data = dataset.dataset[((*partition_conf[part]["timerange"], 3600), ["imerg5625/precipitationcal"], {})][data_idx]
            segger = seg_rain_imerg(data)
            res.append(segger)

        return res


    dataset = Dataset(datapath=datapath,
                      partition_conf=partition_conf,
                      partition_type=partition_type,
                      partition_selected="val",
                      sample_conf=sample_conf,
                      )
    num_idx_shp = dataset.dataset[((*partition_conf[part]["timerange"], 3600), ["imerg5625/precipitationcal"], {})].shape
    num_idx = num_idx_shp[0]
    print("Num idx: {}".format(num_idx))
    n_proc = 60

    idxs = np.array_split(np.array(list(range(num_idx))), n_proc)
    print("IDXS:", idxs)
    with Pool(processes=n_proc) as pool:
        res = pool.map(get_histograms, [(idxlst, i) for idxlst, i in zip(idxs, range(len(idxs)))])

    totres = []
    for r in res:
        totres += r

    with open("histo_{}.json".format(part), "w") as f:
        json.dump(totres, f)
