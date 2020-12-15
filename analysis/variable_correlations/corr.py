from datetime import datetime
import numpy as np
import os, sys
import json
from scipy import stats
from multiprocessing import Pool, TimeoutError
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.memmap_dataloader import Dataset


if __name__ == "__main__":

    # set up dataloader with any dataset type you can think of
    memmap_root = "" # SET MEMMAP DATA ROOT PATH HERE
    datapath = [os.path.join(memmap_root, "simsat5625", "simsat5625.dill"),
                os.path.join(memmap_root, "imerg5625", "imerg5625.dill"),
                os.path.join(memmap_root, "era5625", "era5625.dill"),
                ]

    daterange_train = (datetime(2016, 4, 1).timestamp(), datetime(2017, 12, 31, 23).timestamp())
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

    dr = (datetime(2016, 4, 1).timestamp(), datetime(2019, 12, 31, 21).timestamp())

    def get_corr(args):
        curr_vbl, vbls, x, dr = args
        print ("Starting process {}...".format(x))

        dataset = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="train",
                          sample_conf=sample_conf,
                          )
        res = []
        try:
            dx1 = dataset.dataset[((*dr, 3600*3), [curr_vbl], {})]
        except Exception as e:

            raise Exception("{}: vbl: {} dr: {}".format(e, curr_vbl, dr))
        ds1 = dx1[..., int(dx1.shape[-2]/4):-int(dx1.shape[-2]/4),  int(dx1.shape[-1]/4):-int(dx1.shape[-1]/4)]
        for y, v2 in enumerate(vbls):
            print("Process {}, v1: {} v2: {} it:{} start corr...".format(x, curr_vbl, v2, y))
            dx2 = dataset.dataset[((*dr, 3600*3), [v2], {})]
            ds2 = dx2[..., int(dx2.shape[-2]/4):-int(dx2.shape[-2]/4),  int(dx2.shape[-1]/4):-int(dx2.shape[-1]/4)]
            if len(ds1.shape) < len(ds2.shape):
                print("DONOT: {} shp1: {} shp2: {}".format(np.expand_dims(ds1, axis=0).shape, ds1.shape, ds2.shape))
                try:
                    a =  np.expand_dims(ds1, axis=0).repeat(ds2.shape[0], axis=0)
                except Exception as e:
                    raise Exception("{}: DONOT: {} shp1: {} shp2: {}".format(e, np.expand_dims(ds1, axis=0).shape, ds1.shape, ds2.shape))
                print("WARING: shp1: {} shp2: {} new_shp: {}".format(ds1.shape, ds2.shape, a.shape))
            else:
                a = ds1
            corr = stats.spearmanr(a.flatten(), ds2.flatten())[0] 
            print("Process {}, v1: {} v2: {} it:{} found corr: {}!".format(x, curr_vbl, v2, y, corr))
            res.append(corr)
        return res

    pressure_levels = [300, 500, 850]
    era_lst = ["era5625"]
    simsat_lst = ["simsat5625"]
    imerg_lst = ["imerg5625"] 
    reso_lst = ["5625"] 

    resdct = {}
    for pl in pressure_levels:
        print("pressure level: {}".format(pl))
        resdct[pl] = {}
        for reso, era, simsat, imerg in zip(reso_lst, era_lst, simsat_lst, imerg_lst):

            vbl_list = ['{}/lon2d'.format(era), '{}/lat2d'.format(era), '{}/lsm'.format(era), '{}/orography'.format(era), '{}/slt'.format(era)]
            if era in ["era5625"]:
                vbl_list += ['{}/z_{}hPa'.format(era, pl), '{}/t_{}hPa'.format(era, pl), '{}/q_{}hPa'.format(era, pl),
                           '{}/sp'.format(era), '{}/clwc_{}hPa'.format(era, pl), '{}/ciwc_{}hPa'.format(era, pl), '{}/t2m'.format(era),
                           '{}/clbt:0'.format(simsat), '{}/clbt:1'.format(simsat), '{}/clbt:2'.format(simsat),
                           "{}/tp".format(era), "{}/precipitationcal".format(imerg)]
            else:
                vbl_list += ['{}/z_{}hPa'.format(era, pl), '{}/t_{}hPa'.format(era, pl), '{}/q_{}hPa'.format(era, pl),
                            '{}/sp'.format(era), '{}/clwc_{}hPa'.format(era, pl), '{}/ciwc_{}hPa'.format(era, pl), '{}/t2m'.format(era),
                            '{}/clbt:0'.format(simsat), '{}/clbt:1'.format(simsat), '{}/clbt:2'.format(simsat),
                            "{}/tp".format(era), "{}/precipitationcal".format(imerg)]

            resdct[pl]["vbl_lst"] = vbl_list


            vbl_args = [vbl_list[i:] for i in range(1, len(vbl_list))]
            with Pool(processes=len(vbl_list)) as pool:
                res = pool.map(get_corr, [(v, va, x, dr) for v, va, x in zip(vbl_list[:-1], vbl_args, range(1,len(vbl_list)))])

            try:
                resdct[pl][reso] = np.zeros((len(vbl_list), len(vbl_list)))
            except Exception as e:
                print("res: ", res)
                raise Exception(res[pl].keys())

            for j, r in enumerate(res):
                try:
                    resdct[pl][reso][j, j+1:] = np.array(r)
                except Exception as e:
                    raise Exception(r)
            resdct[pl][reso] = resdct[pl][reso].tolist()


    with open("out.json", "w") as f:
        json.dump(resdct, f)
