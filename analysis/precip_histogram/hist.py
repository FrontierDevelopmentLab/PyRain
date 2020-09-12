from datetime import datetime
import numpy as np
import os, sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataloader.fast_dataloader import Dataset

if __name__ == "__main__":


    # set up dataloader with any dataset type you can think of
    memmap_root = "" # SET MEMMAP DATA ROOT PATH HERE
    datapath = [os.path.join(memmap_root, "simsat5625", "simsat5625.dill"),
                os.path.join(memmap_root, "imerg5625", "imerg5625.dill"),
                os.path.join(memmap_root, "era5625", "era5625.dill"),
                ]

    daterange_train = (datetime(2004, 1, 1).timestamp(), datetime(2009, 12, 31, 23).timestamp())
    daterange_test = (datetime(2019, 1, 6, 0).timestamp(), datetime(2019, 12, 31, 21).timestamp())
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

    dataset = Dataset(datapath=datapath,
                      partition_conf=partition_conf,
                      partition_type=partition_type,
                      partition_selected="train",
                      sample_conf=sample_conf,
                      )

    n_bins = 100

    with open("results/imerg_25bi.json", "r") as f:
        bins = json.load(f)["hist_den"][1]


    # era 5625 histogram
    print("era 5625...")
    era5_tp = dataset.dataset[((*daterange_train, 3600), ["era5625/tp"], {})]
    hist_den = np.histogram(era5_tp.flatten()*1000.0, bins=bins, density=True)
    hist_noden = np.histogram(era5_tp.flatten()*1000.0, bins=bins, density=False)
    res5 = {"hist_den": [x.tolist() for x in hist_den],
            "hist_noden": [x.tolist() for x in hist_noden]}

    with open("./results/era5625.json", "w") as f:
         json.dump(res5, f)

    # era 140625 histogram
    print("era 140625...")
    era1_tp = dataset.dataset[((*daterange_train, 3600), ["era140625/tp"], {})]
    hist_den = np.histogram(era1_tp.flatten()*1000, bins=bins, density=True)
    hist_noden = np.histogram(era1_tp.flatten()*1000, bins=bins, density=False)
    res1 = {"hist_den": [x.tolist() for x in hist_den],
            "hist_noden": [x.tolist() for x in hist_noden]}   

    with open("./results/era140625.json", "w") as f:
         json.dump(res1, f)

    # imerg 140625 histogram
    print("imerg 140625...")
    imerg1_pre = dataset.dataset[((*daterange_train, 3600), ["imerg140625/precipitationcal"], {})]
    hist_den = np.histogram(imerg1_pre.flatten(), bins=bins, density=True)
    hist_noden = np.histogram(imerg1_pre.flatten(), bins=bins, density=False)
    imerg1 = {"hist_den": [x.tolist() for x in hist_den],
              "hist_noden": [x.tolist() for x in hist_noden]} 

    with open("./results/imerg140625.json", "w") as f:
        json.dump(imerg1, f)
    
    print("imerg 5625...")
    imerg5625_pre = dataset.dataset["imerg5625/precipitationcal"]
    hist_den = np.histogram(imerg5625_pre.flatten(), bins=bins, density=True)
    hist_noden = np.histogram(imerg5625_pre.flatten(), bins=bins, density=False)
    imerg5625 = {"hist_den": [x.tolist() for x in hist_den],
               "hist_noden": [x.tolist() for x in hist_noden]}
    with open("./results/imerg5625.json", "w") as f:
        json.dump(imerg5625, f)

