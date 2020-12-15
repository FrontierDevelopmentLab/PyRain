from datetime import datetime, timedelta
import numpy as np
import os
import pickle
import sys
import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataloader.memmap_dataloader import Dataset
from src.benchmark.utils import compute_latitude_weighting, compute_weighted_mse


if __name__ == "__main__":

    memmap_root = "TODO"  # SET MEMMAP DATA ROOT PATH HERE
    memmap_root2 = "TODO"
    datapath = [os.path.join(memmap_root, "imerg_5625", "imerg_5625.dill"),
                os.path.join(memmap_root2, "era5625_mf", "era5625_mf.dill"),
                ]

    daterange_imerg = (datetime(2016, 4, 1), datetime(2017, 12, 31, 21))
    daterange_era = daterange_imerg
    daterange_val = (datetime(2018,1,6,0), datetime(2018, 12,31,23))

    partition_conf = {"era":
                          {"timerange": (daterange_era[0].timestamp(), daterange_era[1].timestamp()),
                           "increment_s": 60 * 60},
                      "imerg":
                          {"timerange": (daterange_imerg[0].timestamp(), daterange_imerg[1].timestamp()),
                           "increment_s": 60 * 60},
                      "val":
                          {"timerange": (daterange_val[0].timestamp(), daterange_val[1].timestamp()),
                           "increment_s": 60 * 60}
                      }

    partition_type = "range"
    sample_conf_era = {"m0": {"era":{"tp_era": {"vbl": "era5625/tp"},"lat2d": {"vbl":"era5625/lat2d"}}}}
    sample_conf_imerg = {"m0": {"era":{"imerg": {"vbl": "imerg5625/precipitationcal"}}}}

    dataset_era = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="era",
                          sample_conf=sample_conf_era,
    )

    grid = dataset_era["era5625/lat2d"]
    lat_grid = compute_latitude_weighting(grid)

    era_dict = {}
    era_dict_ctr = {}
    # calculate weekly climatology for ERA5
    for i, d in enumerate(dataset_era[(daterange_era[0].timestamp(), daterange_era[1].timestamp(), 3600),["era5625/tp"], {}]):
        t = daterange_era[0] + timedelta(seconds=i*3600)
        week = t.isocalendar()[1]
        if week in era_dict_ctr:
            era_dict_ctr[week] += 1
        else:
            era_dict_ctr[week] = 1
        if week in era_dict:
            era_dict[week] += (np.array(d) - era_dict[week]) / float(era_dict_ctr[week])
        else:
            era_dict[week] = np.array(d)
        pass
    print(sorted(era_dict.keys()))
    del dataset_era

    with open("era_climatology.pickle", "wb") as f:
        pickle.dump(era_dict, f)

    dataset_imerg = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="imerg",
                          sample_conf=sample_conf_imerg,
                          )

    imerg_dict = {}
    imerg_dict_ctr = {}
    # calculate weekly climatology for ERA5
    for i, d in enumerate(dataset_imerg[ (daterange_imerg[0].timestamp(), daterange_imerg[1].timestamp(), 3600) , ["imerg5625/precipitationcal"], {}]):
        t = daterange_imerg[0] + timedelta(seconds=i*3600)
        week = t.isocalendar()[1]
        if week in imerg_dict_ctr:
            imerg_dict_ctr[week] += 1
        else:
            imerg_dict_ctr[week] = 1
        if week in imerg_dict:
            imerg_dict[week] += (np.array(d) - imerg_dict[week]) / float(imerg_dict_ctr[week])
        else:
            imerg_dict[week] = np.array(d)
        pass
    print(sorted(imerg_dict.keys()))
    del dataset_imerg

    with open("imerg_climatology.pickle", "wb") as f:
        pickle.dump(era_dict, f)

    ########################## Predict ERA
    dataset_era = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="val",
                          sample_conf=sample_conf_era,
                          )
    re = 0
    for i, d in enumerate(dataset_era[(daterange_val[0].timestamp(), daterange_val[1].timestamp(), 3600), ["era5625/tp"], {}]):
        t = daterange_val[0] + timedelta(seconds=i*3600)
        week = t.isocalendar()[1]
        rms_error = compute_weighted_mse(th.from_numpy(d)*1000, th.from_numpy(era_dict[week])*1000, th.from_numpy(lat_grid))
        rms_error = rms_error**0.5
        re += (rms_error - re) / float(i+1)
        pass
    del dataset_era

    print("ERA RMS:", re)

    ########################## Predict IMERG
    dataset_imerg = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="val",
                          sample_conf=sample_conf_era,
                          )

    re = 0
    for i, d in enumerate(dataset_imerg[(daterange_val[0].timestamp(), daterange_val[1].timestamp(), 1800), ["imerg5625/precipitationcal"], {}]):
        t = daterange_val[0] + timedelta(seconds=i * 1800)
        week = t.isocalendar()[1]
        rms_error = compute_weighted_mse(th.from_numpy(d), th.from_numpy(imerg_dict[week]), th.from_numpy(lat_grid))
        rms_error = rms_error**0.5
        re += (rms_error - re) / float(i+1)
        pass
    del dataset_imerg

    print("IMERG RMS:", re)
