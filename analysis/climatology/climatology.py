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

    memmap_root = "SET_THIS"  # SET MEMMAP DATA ROOT PATH HERE
    memmap_root2 = "SET_THIS"
    datapath = [os.path.join(memmap_root, "imerg_5625", "imerg_5625.dill"),
                os.path.join(memmap_root2, "era5625_mf", "era5625_mf.dill"),
                ]

    daterange_imerg = (datetime(2000, 6, 1,0), datetime(2017, 12, 31, 23))
    daterange_era = (datetime(1979, 1, 1, 7), datetime(2017, 12, 31, 23))

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
    sample_conf_era = {"m0": {"era":{"tp_era": {"vbl": "era5625/tp"},"lat2d": {"vbl":"era5625/lat2d"}}}} # sample modes
    sample_conf_imerg = {"m0": {"era":{"imerg": {"vbl": "imerg5625/precipitationcal"}}}} # sample modes

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

    era_annual_climatology = dataset_era[(daterange_era[0].timestamp(), daterange_era[1].timestamp(), 3600),["era5625/tp"], {}].mean(axis=0)
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

    imerg_annual_climatology = dataset_imerg[ (daterange_imerg[0].timestamp(), daterange_imerg[1].timestamp(), 3600) , ["imerg5625/precipitationcal"], {}].mean(axis=0)
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

    print("ERA WEEKLY RMS:", re)

    re = 0
    for i, d in enumerate(dataset_era[(daterange_val[0].timestamp(), daterange_val[1].timestamp(), 3600), ["era5625/tp"], {}]):
        rms_error = compute_weighted_mse(th.from_numpy(d)*1000, th.from_numpy(era_annual_climatology)*1000, th.from_numpy(lat_grid))
        rms_error = rms_error**0.5
        re += (rms_error - re) / float(i+1)
        pass
    del dataset_era

    print("ERA ANNUAL RMS:", re)

    ########################## Predict IMERG
    dataset_imerg = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="val",
                          sample_conf=sample_conf_era,
                          )

    re = 0
    for i, d in enumerate(dataset_imerg[(daterange_val[0].timestamp(), daterange_val[1].timestamp(), 3600), ["imerg5625/precipitationcal"], {}]):
        t = daterange_val[0] + timedelta(seconds=i * 3600)
        week = t.isocalendar()[1]
        rms_error = compute_weighted_mse(th.from_numpy(d), th.from_numpy(imerg_dict[week]), th.from_numpy(lat_grid))
        rms_error = rms_error**0.5
        re += (rms_error - re) / float(i+1)
        pass

    print("IMERG WEEKLY RMS:", re)

    re = 0
    for i, d in enumerate(dataset_imerg[(daterange_val[0].timestamp(), daterange_val[1].timestamp(), 3600), ["imerg5625/precipitationcal"], {}]):
        rms_error = compute_weighted_mse(th.from_numpy(d), th.from_numpy(imerg_annual_climatology), th.from_numpy(lat_grid))
        rms_error = rms_error**0.5
        re += (rms_error - re) / float(i+1)
        pass
    del dataset_imerg

    print("IMERG ANNUAL RMS:", re)


