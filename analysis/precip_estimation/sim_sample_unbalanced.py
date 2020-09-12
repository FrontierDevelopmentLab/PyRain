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
    with open("./normalise/5625__16-04-01_12:00to17-12-31_11:00.json") as f:
        nl_train = json.load(f)

    nl_train["__const__lon2d"] = {"mean": 0.5, "std":0.28980498288430995}
    nl_train["__const__lat2d"] = {"mean": 0.5, "std":0.29093928798176877}
    nl_train["era5625/slt"] = {"mean": 1.1389103, "std":0.6714027}


    # set up dataloader with any dataset type you can think of
    memmap_root = "" # SET MEMMAP DATA ROOT PATH HERE
    datapath = [os.path.join(memmap_root, "simsat5625", "simsat5625.dill"),
                os.path.join(memmap_root, "imerg5625", "imerg5625.dill"),
                os.path.join(memmap_root, "era5625", "era5625.dill"),
                ]

    daterange_train = (datetime(2016, 4, 1).timestamp(), datetime(2017, 12, 31, 21).timestamp())
    daterange_test = (datetime(2019, 1, 6, 0).timestamp(), datetime(2019, 12, 31, 21).timestamp())
    daterange_val = (datetime(2018, 1, 6, 0).timestamp(), datetime(2018, 12, 31, 21).timestamp())

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

    dlt = 0
    lt = 0
    grid_shape = (32,64)
    sample_conf = {"mode0":  # sample modes
        {
            "sample":  # sample sections
                {
                    "lsm": {"vbl": "era5625/lsm"},  # sample variables
                    "orography": {"vbl": "era5625/orography"},  # sample variables
                    "slt": {"vbl": "era5625/slt"},
                    "__const__lat2d": {"vbl": "__const__lat2d",
                                       "val": np.repeat(np.expand_dims(np.linspace(0.0, 1.0, grid_shape[0]), axis=1),
                                                        grid_shape[1], axis=1)},
                    "__const__lon2d": {"vbl": "__const__lon2d",
                                       "val": np.repeat(np.expand_dims(np.linspace(0.0, 1.0, grid_shape[1]), axis=0),
                                                        grid_shape[0], axis=0)},
                    "clbt:0": {"vbl": "simsat5625/clbt:0",
                                       "t": np.array([dlt]) * 3600,
                                       "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    "clbt:1": {"vbl": "simsat5625/clbt:1",
                                       "t": np.array([dlt]) * 3600,
                                       "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    "clbt:2": {"vbl": "simsat5625/clbt:2",
                                       "t": np.array([dlt]) * 3600,
                                       "interpolate": ["nan", "nearest_past", "nearest_future"][1]}                   
                   
                    },
            "label": {"tp": {"vbl": "imerg5625/precipitationcal",
                             "t": np.array([lt]) * 3600,
                             "interpolate": ["nan", "nearest_past", "nearest_future"][1]}}}
    }

    dr = (datetime(2016, 4, 1).timestamp(), datetime(2019, 12, 31, 21).timestamp())

    part = "train"
    with open("histo_{}.json".format(part), "r") as f:
        histo = np.array(json.load(f))
        histo = histo[slice(None, None, 3)][:-1]
        print("HISTO NOW: ", histo.shape)
        
    histo_trans = histo.transpose()

    n_samples = 250000*4

    from collections import defaultdict
    id_dct = defaultdict(lambda x: [])

    # calc frequencies
    f = []
    for j in range(4):
        fc = np.sum(histo_trans[j])/ float(32*64*histo.shape[0])
        f.append(fc)

    # draw equal number of idxs from each class
    for c in range(4):
        idx_lst = []
        ch = np.random.choice(np.array(list(range(histo.shape[0]))),
                             int(n_samples * f[c] + 0.5),
                             p=histo_trans[c]/np.sum(histo_trans[c]))
        id_dct[c] = ch

    # sort indices by frame
    bcts = []
    for c in range(4):
        print("ch_c: {}".format(id_dct[c]))
        print("minlen: {} max: {}".format(max(id_dct[c]), histo.shape[0]))
        ct = np.bincount(id_dct[c], minlength=histo.shape[0])
        bcts.append(ct)
        print("Bin {} sum: {}".format(c, np.sum(ct)))
    
    print("ID_DCT:", id_dct)
    print("BCTS: ", bcts)
    
    b = np.stack(bcts)
    print("b:", b)

    print("bincount list: {}".format(b))

    # read in every imerg frame and create a rain class histogram for each and save in a file in the end
    def get_pixels(args):
        dataset_indices, frame_idxs, i = args
        print ("Starting process {} indices at iteration {}...".format(len(dataset_indices), i))

        def choose_pixel(coord, frame, c):
            sample = frame
            X = None
            y = None
            latid, lonid = coord
            sample_keys = frame[0]["sample"].keys()
            label_keys = frame[0]["label"].keys()
            sample_lst = []
            for sk in sample_keys:
                if sk[-4:] == "__ts":
                    continue
                s = sample[0]["sample"][sk][...,latid, lonid]
                vn = sample_conf["mode0"]["sample"][sk]["vbl"]
                if sk in ["tp"]:
                    s = np.log(max(s, 0.0)/nl_train[vn]["std"] + 1)
                else:
                    s = (s-nl_train[vn]["mean"])/nl_train[vn]["std"]
                sample_lst.append(s.flatten())
            X = np.concatenate(sample_lst)
            label_lst = []
            for sk in label_keys:
                if sk[-4:] == "__ts":
                    continue
                s = sample[0]["label"][sk][...,latid,lonid]
                vn = sample_conf["mode0"]["label"][sk]["vbl"]
                if sk in ["tp"]:
                   s = np.log(max(s, 0.0) / nl_train[vn]["std"] + 1)
                else:
                   s = (s-nl_train[vn]["mean"])/nl_train[vn]["std"]
                label_lst.append(s.flatten())
            y = np.concatenate(label_lst)

            return X.tolist(), y.tolist(), [c]

        dataset = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected=part,
                          sample_conf=sample_conf,
                          )

        res = []
        for j, frame_idx in enumerate(frame_idxs):

            data_idx = dataset_indices[:, j]

            if not sum(data_idx):
                
                continue


            # compile my own sample

            sam = [{"sample":{}, "label":{}}]

            for k,v in sample_conf["mode0"]["sample"].items():
                if k[:3] == "__c":
                    sam[0]["sample"][k] = v["val"]
                else:
                    g = dataset.dataset[((dr[0], dr[1], 3600), [v["vbl"]], {})]
                    if len(g.shape) == 3:
                        sam[0]["sample"][k] = g
                    else:
                        fidx = frame_idx*3 if k[:4] != "clbt" else frame_idx
                        sam[0]["sample"][k] = dataset.dataset[((*partition_conf[part]["timerange"], 3600), [v["vbl"]], {})][fidx]
            for k,v in sample_conf["mode0"]["label"].items():
                sam[0]["label"][k] = dataset.dataset[((*partition_conf[part]["timerange"], 3600), [v["vbl"]], {})][frame_idx*3]

            frame = sam[0]["label"]["tp"][0

            bounds = [(0.0, 2.5),
                      (2.5, 10.0),
                      (10.0, 50.0),
                      (50.0, 500000.0)]
            for c in range(4):
                # class 0
                idxs = np.where((frame >= bounds[c][0]) & (frame < bounds[c][1]))
                if data_idx[c].size == 0.0:
                    continue

                try:
                    ch = np.random.choice(np.array(list(range(len(idxs[0])))),
                                          data_idx[c])
                except Exception as e:
                    raise Exception("{}: {}, {}".format(e, idxs[0], data_idx[c]))

                if ch.size == 0:
                    continue
                cl = [(idxs[0][h], idxs[1][h]) for h in ch]


                for cl_idx in cl:
                    spl = choose_pixel(cl_idx, sam, c)
                    res.append(spl)

        return res

    n_proc = 40 
    idxs = np.array_split(b, n_proc, axis=1)
    print("IDXS: ", idxs)
    frame_idxs = np.array_split(np.array(range(b[0].shape[0])), n_proc)
    print("FRAMEIDXS: ", frame_idxs)
    with Pool(processes=n_proc) as pool:
        res = pool.map(get_pixels, [(idxlst, fidxs, i) for idxlst, fidxs, i in zip(idxs, frame_idxs, range(len(idxs)))])

    totres = []
    for r in res:
        totres += r

    with open("sim_samples_unb_{}.json".format(part), "w") as f:
        json.dump(totres, f)
