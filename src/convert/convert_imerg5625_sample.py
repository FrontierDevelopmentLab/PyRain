import datetime
from netCDF4 import Dataset as netcdf_Dataset
import os
import numpy as np

if __name__ == "__main__":

    pressure_to_idx = {50:0, 100:1, 150:2, 200:3, 250:4, 300:5, 400:6, 500:7, 600:8, 700:9, 850:10, 925:11, 1000:12}
    idx_to_pressure = {v:k for k,v in pressure_to_idx.items()}
    dataset_range=(datetime.datetime(2000, 6, 1, 0), datetime.datetime(2019, 12, 31, 23))


    years=list(range(2018,2020))
    dataset_name = "imerg5625_sample"
    input_path = "EDIT INPUT PATH TO NETCDF FOLDER"
    output_path = os.path.join("EDIT OUTPUT PATH WHERE MEMMAPS ARE TO BE CREATED", dataset_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    variables_imerg25bi = [
        {"name": "precipitationcal",
         "ftemplate": os.path.join(input_path, "imerg{}{:02d}{:02d}.nc"),
         "dims": (32, 64),
         "levels": list(range(1))},
    ]

    imerg25bi_path = os.path.join(output_path, "{}__imerg5625.mmap".format(dataset_name))
    n_rec_dim = variables_imerg25bi[0]["dims"]
    imerg25bi_sample_freq = 1 
    n_recs = ((datetime.datetime(2019, 12, 31, 23).timestamp()-datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600 ) // imerg25bi_sample_freq + 1
    n_rec_channels = sum([len(vg["levels"]) for vg in variables_imerg25bi])
    dims = (int(n_recs), int(n_rec_channels), *n_rec_dim)
    print("dims: ", dims)
    print("nrecs: ", n_recs)
    imerg25bi_dims = dims
    if os.path.exists(imerg25bi_path):
        print("Skipping iMERG25bi as file exists... ")
    else:
        # write temporal SimSat variables
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        mmap = np.memmap(imerg25bi_path, dtype='float32', mode='w+', shape=dims)

        def write_day(ymd, vbls):
            y, m, d = ymd
            if y < dataset_range[0].year:
                print("iMERG25bi: no data available for year {}".format(y))
                return
            t_offset = int((datetime.datetime(y, m, d, 0).timestamp() - datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600) // imerg25bi_sample_freq
            t_end = int((datetime.datetime(y, m, d, 23).timestamp() - datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600) // imerg25bi_sample_freq + 1
            for i, vbl in enumerate(vbls):
                print("SimSat writing year {} month {} day {} vbl {}...".format(y, m, d, vbl["name"]))
                rootgrp = netcdf_Dataset(os.path.join(input_path, vbl["ftemplate"].format(y, m, d)), "r", format="NETCDF4")
                root_channel = 0 if not i else sum([len(vg["levels"]) for vg in variables_imerg25bi[:i]])
                print(t_offset, t_end, root_channel, len(vbl["levels"]))
                try:
                    mmap[t_offset:t_end, root_channel] = rootgrp[vbl["name"]][:]
                except Exception as e:
                    print(y, m, d, vbl["name"], vbl["levels"], t_offset, t_end, root_channel, e, rootgrp[vbl["name"]][:].shape)
                    raise Exception("{} {} {} {} {} {} {} {} {} {} ".format(y, m, d, vbl["name"], vbl["levels"], t_offset, t_end, root_channel, e, rootgrp[vbl["name"]][:].shape))

        ymd = []
        dd = datetime.datetime(years[0], 1, 1, 0)
        while dd <= datetime.datetime(years[1], 12, 31, 23):
            ymd.append((dd.year, dd.month, dd.day))
            dd += datetime.timedelta(days=1)

        from multiprocessing import Pool
        from functools import partial
        with Pool(40) as p:
            p.map(partial(write_day,vbls=variables_imerg25bi), ymd)
        mmap.flush()
        del mmap


    # Create Pickle file describing which variables are contained in what file at what positions and what frequency
    print("Done converting. Generating dataset pickle file...")
    import dill
    import json
    dct = {}
    dct["variables"] = {}
    for i, v in enumerate(variables_imerg25bi):
        vbl_dict = {"name":v["name"],
                    "mmap_name":"{}__imerg5625.mmap".format(dataset_name),
                    "type":"temp",
                    "dims": v["dims"],
                    "offset": 0 if not i else sum([len(vg["levels"]) for vg in variables_imerg25bi[:i]]),
                    "first_ts": datetime.datetime(years[0], 1, 1, 0).timestamp(),
                    "last_ts": datetime.datetime(years[1], 12, 31, 23).timestamp(),
                    "tfreq_s": 3600,
                    "levels": v["levels"]}
        dct["variables"]["imerg5625/{}".format(v["name"])] = vbl_dict

    dct["memmap"] = {"{}__imerg5625.mmap".format(dataset_name): {"dims": imerg25bi_dims,
                                            "dtype": "float32",
                                            "daterange": (datetime.datetime(years[0], 1, 1, 0).timestamp(),
                                                          datetime.datetime(years[1], 12, 31, 23).timestamp()),
                                            "tfreq_s": 3600}
                     }

    dill.dump(dct, open(os.path.join(output_path, dataset_name+".dill"),'wb'))

    with open(os.path.join(output_path, dataset_name+"_info.json"), 'w') as outfile:
        json.dump(dct, outfile, indent=4, sort_keys=True)
