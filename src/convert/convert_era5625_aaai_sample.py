import datetime
from netCDF4 import Dataset as netcdf_Dataset
import os
import numpy as np

if __name__ == "__main__":

    pressure_to_idx = {50:0, 100:1, 150:2, 200:3, 250:4, 300:5, 400:6, 500:7, 600:8, 700:9, 850:10, 925:11, 1000:12}
    idx_to_pressure = {v:k for k,v in pressure_to_idx.items()}

    years=list(range(2018,2020))
    dataset_name = "era5625_sample"
    input_path = "EDIT INPUT PATH TO NETCDF FOLDER"
    output_path = os.path.join("EDIT OUTPUT PATH WHERE MEMMAPS ARE TO BE CREATED", dataset_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    variables_const = [
        {"name": "lat2d",
         "ftemplate": os.path.join(input_path, "constants_5.625deg.nc"),
         "dims": (32, 64)},
        {"name": "lon2d",
         "ftemplate": os.path.join(input_path, "constants_5.625deg.nc"),
         "dims": (32, 64)},
        {"name": "lsm",
         "ftemplate": os.path.join(input_path, "constants_5.625deg.nc"),
         "dims": (32, 64)},
        {"name": "orography",
         "ftemplate": os.path.join(input_path, "constants_5.625deg.nc"),
         "dims": (32, 64)},
        {"name": "slt",
         "ftemplate": os.path.join(input_path, "constants_5.625deg.nc"),
         "dims": (32, 64)},
            ]

    variables_era = [
        {"name": "t2m",
         "ftemplate": os.path.join(input_path, "2m_temperature_{}_5.625deg.nc"),
         "dims": (32, 64),
         "levels": list(range(1))},
        {"name": "sp",
         "ftemplate": os.path.join(input_path, "surface_pressure_{}_5.625deg.nc"),
         "dims": (32, 64),
         "levels": list(range(1))},
        {"name": "tp",
         "ftemplate": os.path.join(input_path, "total_precipitation_{}_5.625deg.nc"),
         "dims": (32, 64),
         "levels": list(range(1))}, #]#,
    ]

    from copy import deepcopy
    variables_era_2019 = deepcopy(variables_era)

    # era_extra_pressure_levels = [300, 500, 850] #, 850]
    # for i, p in enumerate(era_extra_pressure_levels):
    #     variables_era.append({"name": "ciwc".format(p),
    #                           "ftemplate": os.path.join(input_path, "specific_cloud_ice_water_content_{}_"+str(int(p))+"_5.625deg.nc"),
    #                           "dims": (32, 64),
    #                           "levels": list((pressure_to_idx[p],)),
    #                           "p_level": p})
    #     variables_era.append({"name": "clwc".format(p),
    #                           "ftemplate": os.path.join(input_path, "specific_cloud_liquid_water_content_{}_"+str(int(p))+"_5.625deg.nc"),
    #                           "dims": (32, 64),
    #                           "levels": list((pressure_to_idx[p],)),
    #                           "p_level": p})
    #     variables_era.append({"name": "t",
    #                           "ftemplate": os.path.join(input_path, "temperature_{}_5.625deg.nc"),
    #                           "dims": (32, 64),
    #                           "levels": list((pressure_to_idx[p],)),
    #                           "p_level": p}),
    #     variables_era.append({"name": "z",
    #                          "ftemplate": os.path.join(input_path, "geopotential_{}_5.625deg.nc"),
    #                          "dims": (32, 64),
    #                          "levels": list((pressure_to_idx[p],))}),
    #     variables_era.append({"name": "q",
    #                          "ftemplate": os.path.join(input_path, "specific_humidity_{}_5.625deg.nc"),
    #                          "dims": (32, 64),
    #                          "levels": list((pressure_to_idx[p],))}),

    # era_extra_pressure_levels = [300, 500, 850] #, 850]
    # for i, p in enumerate(era_extra_pressure_levels):
    #     variables_era_2019.append({"name": "ciwc".format(p),
    #                           "ftemplate": os.path.join(input_path, "specific_cloud_ice_water_content_{}_"+str(int(p))+"_5.625deg.nc"),
    #                           "dims": (32, 64),
    #                           "levels": list((pressure_to_idx[p],)),
    #                           "p_level": p})
    #     variables_era_2019.append({"name": "clwc".format(p),
    #                           "ftemplate": os.path.join(input_path, "specific_cloud_liquid_water_content_{}_"+str(int(p))+"_5.625deg.nc"),
    #                           "dims": (32, 64),
    #                           "levels": list((pressure_to_idx[p],)),
    #                           "p_level": p})
    #     variables_era_2019.append({"name": "t",
    #                           "ftemplate": os.path.join(input_path, "temperature_{}_"+str(int(p))+"_5.625deg.nc"),
    #                           "dims": (32, 64),
    #                           "levels": list((pressure_to_idx[p],)),
    #                           "p_level": p}),
    #     variables_era_2019.append({"name": "z",
    #                          "ftemplate": os.path.join(input_path, "geopotential_{}_"+str(int(p))+"_5.625deg.nc"),
    #                          "dims": (32, 64),
    #                          "levels": list((pressure_to_idx[p],))}),
    #     variables_era_2019.append({"name": "q",
    #                          "ftemplate": os.path.join(input_path, "specific_humidity_{}_"+str(int(p))+"_5.625deg.nc"),
    #                          "dims": (32, 64),
    #                          "levels": list((pressure_to_idx[p],))}),

    era_const_path = os.path.join(output_path, "{}__era5625_const.mmap".format(dataset_name))
    print("Writing const values...")
    const_dims = (sum([1 for vg in variables_const]), 32, 64)
    era_const_dims = const_dims

    if os.path.exists(era_const_path):
        print("Skipping ERA CONST as file exists... ")
    else:
        # write const variables
        mmap = np.memmap(era_const_path, dtype='float32', mode='w+', shape=const_dims)
        def write_const(vbls):
            rootgrp = netcdf_Dataset(os.path.join(input_path, vbls[0]["ftemplate"]), "r", format="NETCDF4")
            for i, vbl in enumerate(vbls):
                print("WRITING CONST VBL ", vbl["name"])
                root_channel = 0 if not i else sum([1 for vg in variables_const[:i]])
                print("ROOT CHANNEL: ", root_channel)
                mmap[root_channel] = rootgrp[vbl["name"]][:]
        write_const(variables_const)
        mmap.flush()
        del mmap

    # write temporal ERA variables
    n_rec_dim = (32, 64)
    n_recs = (datetime.datetime(min(max(years), 2019), 12, 31, 23).timestamp()-datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600 + 1
    n_rec_channels = sum([len(vg["levels"]) for vg in variables_era])
    dims = (int(n_recs), int(n_rec_channels), *n_rec_dim)
    era_dims = dims
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    era_path = os.path.join(output_path, "{}__era5625.mmap".format(dataset_name))
    if os.path.exists(era_path):
        print("Skipping ERA as file exists... ")
    else:
        mmap = np.memmap(era_path, dtype='float32', mode='w+', shape=dims)

        def write_year(y, vbls):
            vbls, vbls_2019 = vbls

            if y > 2019:
                print("ERA: no data available for year {}".format(y))
                return
            t_offset = int((datetime.datetime(y, 1, 1, 0).timestamp() - datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600)
            t_end = int((datetime.datetime(y, 12, 31, 23).timestamp() - datetime.datetime(years[0], 1, 1, 0).timestamp()) // 3600) + 1
            for i, vbl in enumerate(vbls):
                if y == 2019:
                    vbl = vbls_2019[i]
                print("ERA5625 writing year {} vbl {}...".format(y, vbl["name"]))
                netcdf_fname = vbl["ftemplate"].format(y)
                root_channel = 0 if not i else sum([len(vg["levels"]) for vg in variables_era[:i]])
                if vbl["name"] in ["tcwv"] and y > 2000:
                    mmap[t_offset:t_end, root_channel] = float("nan")
                else:
                    rootgrp = netcdf_Dataset(os.path.join(input_path, netcdf_fname), "r", format="NETCDF4")
                    print(t_offset, t_end, root_channel, len(vbl["levels"]))
                    if vbl["name"] in ["tisr", "tp"] and y == 1979:
                        mmap[t_offset+7:t_end, root_channel] = rootgrp[vbl["name"]][:] # tisr, tp starts at 7:00 o clock
                        mmap[t_offset:t_offset+7, root_channel] = float("nan")
                    else:
 
                        if len(vbl["levels"]) == 1:
                            mmap[t_offset:t_end, root_channel] = rootgrp[vbl["name"]][:] #[:, vbl["levels"]]
                        else:
                            mmap[t_offset:t_end, root_channel:root_channel+len(vbl["levels"])] = rootgrp[vbl["name"]][:, vbl["levels"]]
                #mmap.flush()

        from multiprocessing import Pool
        from functools import partial
        with Pool(40) as p:
            p.map(partial(write_year,vbls=(variables_era, variables_era_2019)), years)
        mmap.flush()
        del mmap


    # Create Pickle file describing which variables are contained in what file at what positions and what frequency
    print("Done converting. Generating dataset pickle file...")
    import dill
    import json
    dct = {}
    dct["variables"] = {}
    for i, v in enumerate(variables_const):
        vbl_dict = {"name":v["name"],
                    "mmap_name":"{}__era5625_const.mmap".format(dataset_name),
                    "type":"const",
                    "dims": v["dims"],
                    "offset": 0 if not i else sum([1 for vg in variables_const[:i]]),
                    "first_ts": None,
                    "last_ts": None,
                    "tfreq_s": None,
                    "levels": None}
        dct["variables"]["era5625/{}".format(v["name"])] = vbl_dict

    for i, v in enumerate(variables_era):
        vbl_dict = {"name": "{}_{}hPa".format(v["name"], v["p_level"]) if v["name"] in ["ciwc","clwc"] else v["name"],
                    "mmap_name":"{}__era5625.mmap".format(dataset_name),
                    "type":"temp",
                    "dims": v["dims"],
                    "offset": 0 if not i else sum([len(vg["levels"]) for vg in variables_era[:i]]),
                    "first_ts": datetime.datetime(years[0], 1, 1, 0).timestamp(),
                    "last_ts": datetime.datetime(years[1], 12, 31, 23).timestamp(),# if v["name"] not in ["ciwc", "clwc"] else datetime.datetime(2000,12,31,23).timestamp(),
                    "tfreq_s": 3600,
                    "levels": v["levels"]}

        if "p_level" in v: 
            vbl_dict["index2pressure"] = {i:int(v["p_level"]) for i, vl in enumerate(v["levels"])}
        else:
            vbl_dict["index2pressure"] = {i:int(idx_to_pressure[vl]) for i, vl in enumerate(v["levels"])}
        dct["variables"]["era5625/{}".format(vbl_dict["name"])] = vbl_dict

    dct["memmap"] = {"{}__era5625_const.mmap".format(dataset_name): {"dims": era_const_dims,
                                            "dtype": "float32",
                                            "daterange": None,
                                            "tfreq_s": None},
                     "{}__era5625.mmap".format(dataset_name): {"dims": era_dims,
                                      "dtype": "float32",
                                      "daterange": (datetime.datetime(years[0], 1, 1, 0).timestamp(), datetime.datetime(years[1], 12, 31, 23).timestamp()),
                                      "tfreq_s": 3600},
                     }

    dill.dump(dct, open(os.path.join(output_path, dataset_name+".dill"),'wb'))

    with open(os.path.join(output_path, dataset_name+"_info.json"), 'w') as outfile:
        json.dump(dct, outfile, indent=4, sort_keys=True)
