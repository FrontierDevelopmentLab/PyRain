import datetime
from netCDF4 import Dataset as netcdf_Dataset
import os
import numpy as np

if __name__ == "__main__":

    pressure_to_idx = {50:0, 100:1, 150:2, 200:3, 250:4, 300:5, 400:6, 500:7, 600:8, 700:9, 850:10, 925:11, 1000:12}
    idx_to_pressure = {v:k for k,v in pressure_to_idx.items()}

    years=list(range(2018,2020))
    dataset_name = "simsat5625"
    input_path = "EDIT INPUT PATH TO NETCDF FOLDER"
    output_path = os.path.join("EDIT OUTPUT PATH WHERE MEMMAPS ARE TO BE CREATED", dataset_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    variables_simsat = [
        {"name": "clbt",
         "ftemplate": os.path.join(input_path, "sat{}.nc"),
         "dims": (32, 64),
         "levels": list(range(3))},
    ]

    ds_daterange = (datetime.datetime(2016, 4, 1, 0), datetime.datetime(2020, 3, 31, 21)) 
    ts_daterange = ds_daterange

    simsat_path = os.path.join(output_path, "{}__simsat5625.mmap".format(dataset_name))
    n_rec_dim = (32, 64)
    simsat_sample_freq = 3 # every 3 hours
    n_recs = ((datetime.datetime(2019, 12, 31, 23).timestamp()-datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600 ) // simsat_sample_freq + 1
    n_rec_channels = sum([len(vg["levels"]) for vg in variables_simsat])
    dims = (int(n_recs), int(n_rec_channels), *n_rec_dim)
    simsat_dims = dims
    if os.path.exists(simsat_path):
        print("Skipping SimSat as file exists... ")
    else:
        # write temporal SimSat variables
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        mmap = np.memmap(simsat_path, dtype='float32', mode='w+', shape=dims)
        print("MMAP DIMS: ", dims)

        def write_year(y, vbls):
            if y < 2016:
                print("SimSat: no data available for year {}".format(y))
                return
            if y == 2016:
                t_offset = 0
            else:
                t_offset = int((datetime.datetime(y,1,1,0).timestamp() - ds_daterange[0].timestamp()) // 3600) // simsat_sample_freq
            if y == 2020:
                t_end = int((ts_daterange[1].timestamp() - ds_daterange[0].timestamp()) // 3600) // simsat_sample_freq + 1
            else:
                t_end = int((datetime.datetime(y, 12, 31, 23).timestamp() - datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600) // simsat_sample_freq + 1
            print("year: ", y, " t_offset: ", t_offset, "t_end:", t_end)
            for i, vbl in enumerate(vbls):
                print("SimSat writing year {} vbl {}...".format(y, vbl["name"]))
                rootgrp = netcdf_Dataset(os.path.join(input_path, vbl["ftemplate"].format(y)), "r", format="NETCDF4")
                root_channel = 0 if not i else sum([len(vg["levels"]) for vg in variables_simsat[:i]])
                print("hello:", t_offset, t_end, root_channel, len(vbl["levels"]), rootgrp[vbl["name"]].shape)
                try:
                    mmap[t_offset:t_end, root_channel:root_channel+len(vbl["levels"])] = rootgrp[vbl["name"]][:, vbl["levels"]]
                except Exception as e:
                    print("EXCEPTION", rootgrp[vbl["name"]].shape, t_offset, t_end, root_channel, mmap.shape)
                    raise Exception()

        from multiprocessing import Pool
        from functools import partial
        with Pool(1) as p:
            p.map(partial(write_year,vbls=variables_simsat), years)
        mmap.flush()
        del mmap


    # Create Pickle file describing which variables are contained in what file at what positions and what frequency
    print("Done converting. Generating dataset pickle file...")
    import dill
    import json
    dct = {}
    dct["variables"] = {}
    for i, v in enumerate(variables_simsat):
        vbl_dict = {"name":v["name"],
                    "mmap_name":"{}__simsat5625.mmap".format(dataset_name),
                    "type":"temp",
                    "dims": v["dims"],
                    "offset": 0 if not i else sum([len(vg["levels"]) for vg in variables_simsat[:i]]),
                    "first_ts": datetime.datetime(years[0], 1, 1, 0).timestamp(),
                    "last_ts": datetime.datetime(years[1], 12, 31, 23).timestamp(),
                    "tfreq_s": 3600*3,
                    "levels": v["levels"]}
        dct["variables"]["simsat5625/{}".format(v["name"])] = vbl_dict

    dct["memmap"] = {"{}__simsat5625.mmap".format(dataset_name): {"dims": simsat_dims,
                                            "dtype": "float32",
                                            "daterange": (datetime.datetime(years[0], 1, 1, 0).timestamp(),
                                                          datetime.datetime(years[1], 12, 31, 23).timestamp()),
                                            "tfreq_s": 3600*3}
                     }

    dill.dump(dct, open(os.path.join(output_path, dataset_name+".dill"),'wb'))

    with open(os.path.join(output_path, dataset_name+"_info.json"), 'w') as outfile:
        json.dump(dct, outfile, indent=4, sort_keys=True)
