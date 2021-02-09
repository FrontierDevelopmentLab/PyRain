import bisect
import dill
from datetime import datetime
import math
import numpy as np
import torch
import os
from collections import defaultdict
from tqdm import tqdm
from haversine import haversine
import torch as th
from multiprocessing import Pool
from functools import partial, reduce

from .utils import haversine_distance as hv_d
from contextlib import closing
import multiprocessing as mp
from operator import mul

from google.cloud import storage

from scipy.ndimage import gaussian_filter1d


# sharing numpy array in multiprocessing pipeline taken from: https://gist.github.com/rossant/7a46c18601a2577ac527f958dd4e452f

def _init(arrs):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    mean_shared_arr_, std_shared_arr_ = arrs
    global mean_shared_arr, std_shared_arr
    mean_shared_arr = mean_shared_arr_
    std_shared_arr = std_shared_arr_


def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    shared_arr = mp.RawArray(cdtype, reduce(mul, shape, 1))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr


def _normalisation_lalas(coords, vbl_arr, vbl_indices, vbl_dims, kernel_dim, vbl_full_type=None, mean_arr=None,
                         std_arr=None):
    if mean_arr is None or std_arr is None:
        mean_arr = shared_to_numpy(*mean_shared_arr)
        std_arr = shared_to_numpy(*std_shared_arr)

    lon_idx, lat_idx, level_idx = coords
    wsize_0 = kernel_dim  # kernel size at 0 latitude
    dist_0 = haversine((0, 0), (0, wsize_0))
    lat_deg_inc = (1.0 / vbl_dims[-2]) * 180.0
    lat_deg = (lat_idx - vbl_dims[-2] / 2.0) * lat_deg_inc
    eff_wsize = min(math.ceil(wsize_0 * dist_0 / (haversine((lat_deg, 0),
                                                            (lat_deg, wsize_0))) + 10E-14), 2 * vbl_dims[0])
    print("LALAS normalisation - vbl_type: {}, lon: {}, lat: {} level: {}".format(vbl_full_type,
                                                                                  lon_idx,
                                                                                  lat_idx,
                                                                                  level_idx))
    if level_idx is None:
        idxs = np.ravel_multi_index(np.ix_(vbl_indices,
                                           np.arange(lat_idx, min(lat_idx + eff_wsize, mean_arr.shape[0])),
                                           np.arange(lon_idx, lon_idx + wsize_0)),
                                    vbl_arr.shape,
                                    mode="wrap")
        mean_arr[lat_idx, lon_idx] = np.mean(np.take(vbl_arr, idxs))
        std_arr[lat_idx, lon_idx] = np.std(np.take(vbl_arr, idxs))
    else:
        idxs = np.ravel_multi_index(np.ix_(vbl_indices,
                                           [level_idx],
                                           np.arange(lat_idx, min(lat_idx + eff_wsize, mean_arr.shape[0])),
                                           np.arange(lon_idx, lon_idx + wsize_0)),
                                    vbl_arr.shape,
                                    mode="wrap")
        mean_arr[level_idx, lat_idx, lon_idx] = np.mean(np.take(vbl_arr, idxs))
        std_arr[level_idx, lat_idx, lon_idx] = np.std(np.take(vbl_arr, idxs))
    return mean_arr, std_arr


def normalisation_lalas(vbl_arr, vbl_indices, vbl_dims, vbl_levels, kernel_dim, num_workers=10, vbl_full_type=None):
    # latitude-adjusted LAS

    if vbl_levels is None:
        # mean = np.zeros(vbl_dims)
        # std = np.zeros_like(mean)
        mean_shared_arr, mean_arr = create_shared_array(np.float32, vbl_dims)
        std_shared_arr, std_arr = create_shared_array(np.float32, mean_arr.shape)
        p_func = partial(_normalisation_lalas,
                         # vbl_arr=vbl_arr,
                         vbl_indices=vbl_indices,
                         vbl_dims=vbl_dims,
                         kernel_dim=kernel_dim,
                         vbl_full_type=vbl_full_type
                         )
        coords = [(a0, b0, None) for a0 in range(vbl_dims[1]) for b0 in range(vbl_dims[0] + 1)]

    else:
        mean_shared_arr, mean_arr = create_shared_array(np.float32, (vbl_levels, *vbl_dims))
        std_shared_arr, std_arr = create_shared_array(np.float32, mean_arr.shape)
        p_func = partial(_normalisation_lalas,
                         # vbl_arr=vbl_arr,
                         vbl_indices=vbl_indices,
                         vbl_dims=vbl_dims,
                         kernel_dim=kernel_dim,
                         vbl_full_type=vbl_full_type
                         )
        coords = [(a0, b0, c0) for a0 in range(vbl_dims[1]) for b0 in range(vbl_dims[0]) for c0 in range(vbl_levels)]

    if num_workers:
        with closing(mp.Pool(
                num_workers, initializer=_init, initargs=(((mean_shared_arr, np.float32, mean_arr.shape),
                                                           (std_shared_arr, np.float32, std_arr.shape)),))) as p:
            p.map(p_func, coords)
    else:
        meaner = None
        stder = None
        for coord in coords:
            m, s = _normalisation_lalas(coords=coord,
                                        vbl_arr=vbl_arr,
                                        vbl_indices=vbl_indices,
                                        vbl_dims=vbl_dims,
                                        kernel_dim=kernel_dim,
                                        vbl_full_type=vbl_full_type,
                                        mean_arr=mean_arr,
                                        std_arr=std_arr)
            if meaner is None:
                meaner = m
            else:
                meaner += m
            if stder is None:
                stder = s
            else:
                stder += s
        mean_arr = np.mean(meaner)
        std_arr = np.std(stder)

    # with Pool(num_workers) as p:
    #    p.map(p_func, coords)

    mean_arr = gaussian_filter1d(mean_arr, sigma=10, axis=-1, mode='wrap')  # wrap around as it it is the longitude
    mean_arr = gaussian_filter1d(mean_arr, sigma=10, axis=-2, mode='nearest')
    std_arr = gaussian_filter1d(std_arr, sigma=10, axis=-1, mode='wrap')  # wrap around as it it is the longitude
    std_arr = gaussian_filter1d(std_arr, sigma=10, axis=-2, mode='nearest')

    return {"mean": mean_arr,
            "std": std_arr}


class Normalise(object):
    """
    Normalise samples across a given kernel size.
    Mean and Std are calculated across time across a (weighted) kernel
    """

    def __init__(self, dataset, normalisation_dict=None, bucket_name="fdl_dte_normalisation", num_workers=8):
        self.dataset = dataset
        self.sample_config = dataset.sample_conf
        self.default_normalisation_dict = {"__temp": "lalas:kernel={}".format(7),
                                           "__const": "framewise",
                                           "yera5625/tp": "lalas@precip:kernel={}".format(7),
                                           "era5625/tp": "lalas@precip:kernel={}".format(7),
                                           "era140625/tp": "lalas@precip:kernel={}".format(7),
                                           "era025/tp": "lalas@precip:kernel={}".format(7)}
        self.normalisation_dict = {}
        self.normalisation_dict.update(self.default_normalisation_dict)
        if normalisation_dict is not None:
            self.normalisation_dict.update(normalisation_dict)

        self.bucket_name = bucket_name
        self.num_workers = num_workers
        self.build_normalisation_cache()
        pass

    def _make_uid(self, dct):

        nt = dct["normalisation_str"].split(":")
        nc = [] if len(nt) <= 1 else nt[1]

        nt = nt[0].split("@")
        ac = None if len(nt[0].split("@")) <= 1 else nt[0].split("@")[1]
        if len(nc):
            norm_str = "{}:{}".format(nt[0], nc)
        else:
            norm_str = nt[0]

        uid_str = "type:{}_norm:{}".format(dct["vbl_full_type"], norm_str)
        if dct["vbl_is_temp"]:
            if dct["partition_type"] in ["range"]:
                uid_str += "_range:({},{})".format(
                    datetime.fromtimestamp(dct["timerange"][0]).strftime("%m-%d-%Y,%H.%M.%S"),
                    datetime.fromtimestamp(dct["timerange"][1]).strftime("%m-%d-%Y,%H.%M.%S"),
                    )
            elif dct["partition_type"] in ["repeat"]:
                uid_str += "__repeat({}to{})_lens:{}_offsets:{}".format(
                    datetime.fromtimestamp(dct["timerange"][0]).strftime("%m-%d-%Y,%H.%M.%S"),
                    datetime.fromtimestamp(dct["timerange"][1]).strftime("%m-%d-%Y,%H.%M.%S"),
                    dct["len_s"],
                    dct["offset_s"]
                    )
            else:
                raise Exception("Unknown partition type! {}".format(dct["partition"]))
        else:
            pass
        return uid_str

    def _get_bucket_cache(self, blob_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        stats = storage.Blob(bucket=bucket, name=blob_name).exists(storage_client)
        blob = bucket.get_blob(blob_name)
        return None if not stats else dill.loads(blob.download_as_string())

    def _set_bucket_cache(self, blob_name, content):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        import io
        f = io.BytesIO(dill.dumps(content))
        blob.upload_from_file(f)

    def build_normalisation_cache(self):

        self.normalisation_cache = {}

        # Go through all variables in the sample config and check whether their required normalisation already exists
        # across all partitions
        to_calculate = {}

        for mode_k, mode_v in self.sample_config.items():
            for section_k, section_v in mode_v.items():
                # for section_k, section_v in section_v.items():
                for vbl_k, vbl_v in section_v.items():
                    print("Assessing mode_k: {}, section_k: {} and vbl_k: {}".format(mode_k, section_k, vbl_k))
                    vbl_name = vbl_k
                    vbl_type = vbl_v["vbl"].split(":")[0]
                    vbl_full_type = vbl_v["vbl"]
                    vbl_grid_dims = self.dataset.dataset.dataset_config["variables"][vbl_type]["dims"]
                    vbl_level_dim = self.dataset[vbl_v["vbl"]].shape[-3]
                    vbl_is_temp = self.dataset.dataset.dataset_config["variables"][vbl_type].get("type", None) == "temp"
                    dct_0 = {"vbl_type": vbl_type,
                             "vbl_full_type": vbl_full_type,
                             "vbl_name": vbl_name,
                             "vbl_dims": vbl_grid_dims,
                             "vbl_levels": vbl_level_dim,
                             "vbl_is_temp": vbl_is_temp}
                    if "normalisation_str" in vbl_v:
                        ns = vbl_v["normalisation_str"]
                    elif "{}__{}__{}".format(mode_k, section_k, vbl_name) in self.normalisation_dict:
                        ns = self.normalisation_dict["{}__{}__{}".format(mode_k, section_k, vbl_name)]
                    elif "{}__{}".format(section_k, vbl_name) in self.normalisation_dict:
                        ns = self.normalisation_dict["{}__{}".format(section_k, vbl_name)]
                    elif "{}".format(vbl_name) in self.normalisation_dict:
                        ns = self.normalisation_dict["{}".format(vbl_name)]
                    elif vbl_full_type in self.normalisation_dict:
                        ns = self.normalisation_dict[vbl_full_type]
                    elif vbl_type in self.normalisation_dict:
                        ns = self.normalisation_dict[vbl_type]
                    elif vbl_is_temp:
                        ns = self.normalisation_dict["__temp"]
                    else:
                        ns = self.normalisation_dict["__const"]

                    dct_lst = []
                    if vbl_is_temp:
                        if self.dataset.partition_type == "range":
                            for partition_k, partition_v in self.dataset.partition_conf.items():
                                print("Get partition segment {}...".format(partition_k))
                                dct = {}
                                dct.update({"timerange": partition_v["timerange"],
                                            "increment_s": partition_v["increment_s"],
                                            "partition_type": "range",
                                            "normalisation_str": ns})
                                # retrieve dataset indices
                                print("Get dataset indices...")
                                rg = self.dataset.get_partition_ts_segments(partition_k)
                                print("Retrieved range: {}".format(rg))
                                idx_rg = self.dataset.get_file_indices_from_ts_range(rg[0], vbl_type)
                                dct["indices"] = list(range(idx_rg[0], idx_rg[1] + 1))
                                dct_lst.append(dct)

                        elif self.dataset.partition_type == "repeat":
                            for i, partition in enumerate(self.dataset.partition_conf["partitions"]):
                                dct = {}
                                dct.update({"timerange": self.dataset.partition_conf["timerange"],
                                            "len_s": partition["len_s"],
                                            "offset_s": sum([p["len_s"] for p in
                                                             self.dataset.partition_conf["partitions"][
                                                             :i]]) if i else 0,
                                            "increment_s": partition["increment_s"],
                                            "partition_type": "repeat",
                                            "normalisation_str": ns})
                                # retrieve dataset indices
                                ts_segs = self.dataset.get_partition_ts_segments(partition["name"])
                                unique_idx = set()
                                for ts_seg in ts_segs:
                                    rg = self.dataset.get_file_indices_from_ts_range(ts_seg, vbl_type)
                                    unique_idx.update(list(range(rg[0], rg[1] + 1)))
                                dct["indices"] = sorted(unique_idx)
                                dct_lst.append(dct)
                        else:
                            raise Exception("Unknown partition type! {}".format(self.dataset.partition_type))
                    else:
                        dct_0.update({"normalisation_str": ns})
                    print("Going through dct list....")
                    for dct in (dct_lst if dct_lst != [] else [{}]):
                        dct.update(dct_0)
                        uid = self._make_uid(dct)
                        bucket_cache = self._get_bucket_cache(uid)
                        key = (mode_k, section_k, vbl_k)
                        if bucket_cache is None:
                            if uid in to_calculate:
                                to_calculate[uid][0].append(key)
                            else:
                                to_calculate[uid] = [key], dct
                        else:
                            self.normalisation_cache[key] = {"normalisation_str": ns,
                                                             "cache": bucket_cache}

        # NOTE: Here we implement a lame, single-process version for now!
        for k, item in to_calculate.items():
            print("Calculating item: {}".format(k))
            stats = self.get_normalisation_stats(item[1]["normalisation_str"],
                                                 item[1]["vbl_full_type"],
                                                 item[1]["vbl_dims"],
                                                 item[1].get("vbl_levels", None),
                                                 item[1]["indices"] if item[1]["vbl_is_temp"] else None)
            for _item in item[0]:
                self.normalisation_cache[_item] = stats
            self._set_bucket_cache(k, stats)

        pass

    def get_normalisation_stats(self, ns, vbl_full_type, vbl_dims, vbl_levels, indices, dataset=None):
        if dataset is None:
            dataset = self.dataset
        nt = ns.split(":")
        if len(nt) > 1:
            nc = {item.split("=")[0]: item.split("=")[1] for item in nt[1].split(";")}
        else:
            nc = {}

        splt = nt[0].split("@")
        ac = None if len(splt) <= 1 else splt[1]
        nt = splt[0]

        if nt in ["lalas"]:
            print("Starting lalas normalisation...")
            # ret =  normalisation_lalas(self.dataset[vbl_full_type],
            #                            vbl_indices=indices,
            #                            vbl_dims=vbl_dims,
            #                            vbl_levels=vbl_levels,
            #                            kernel_dim=int(nc.get("kernel", 7)),
            #                            num_workers=self.num_workers,
            #                            vbl_full_type=vbl_full_type)
            nsplits = len(indices) // 1000
            split_idxs = np.split(nsplits)
            std = None
            mean = None
            for sidx in split_idxs:
                if not mean:
                    mean = np.mean(self.dataset[vbl_full_type][sidx])
                else:
                    mean += np.mean(self.dataset[vbl_full_type][sidx])
                if not std:
                    std = np.std(self.dataset[vbl_full_type][sidx])
                else:
                    std += np.std(self.dataset[vbl_full_type][sidx])
            mean = np.mean(mean)
            std = np.mean(std)

            return ret

        elif nt in ["global"]:
            print("Starting global normalisation for {}".format(vbl_full_type))
            if indices is not None:
                print("DBGU:", self.dataset[vbl_full_type][indices].shape)
                mean = np.mean(self.dataset[vbl_full_type][indices], axis=(0, 2, 3))
                std = np.std(self.dataset[vbl_full_type][indices], axis=(0, 2, 3))
            else:
                mean = np.mean(self.dataset[vbl_full_type])
                std = np.std(self.dataset[vbl_full_type])

            # nsplits = len(indices) // 1000
            # split_idxs = np.split(nsplits)
            # std = None
            # mean = None
            # for sidx in split_idxs:
            #    if not mean:
            #        mean = np.mean(self.dataset[vbl_full_type][sidx])
            #    else:
            #        mean += np.mean(self.dataset[vbl_full_type][sidx])
            #    if not std:
            #        std = np.std(self.dataset[vbl_full_type][sidx])
            #    else:
            #        std += np.std(self.dataset[vbl_full_type][sidx])
            # mean = np.mean(mean)
            # std = np.mean(std)
            return {"mean": mean,
                    "std": std}
        elif nt in ["las"]:
            #  LAS based on:
            #  https://github.com/spcl/deep-weather/blob/master/Uncertainty_Quantification/Preprocessing/npy2tfr.py

            from scipy.ndimage import gaussian_filter1d
            wsize = nc.get("kernel", 7)
            if vbl_levels is None:
                mean = np.zeros(vbl_dims)
                std = np.zeros_like(mean)
                for lon_idx in range(vbl_dims[1] - wsize + 1):
                    eff_wsize = wsize
                    for lat_idx in range(vbl_dims[0] - eff_wsize + 1):
                        mean[lat_idx, lon_idx] = np.mean(self.dataset[vbl_full_type][indices,
                                                         lat_idx:lat_idx + eff_wsize, lon_idx:lon_idx + eff_wsize],
                                                         axis=(0, -2, -1))
                        std[lat_idx, lon_idx] = np.mean(self.dataset[vbl_full_type][indices,
                                                        lat_idx:lat_idx + eff_wsize, lon_idx:lon_idx + eff_wsize],
                                                        axis=(0, -2, -1))

                # change dimension back py padding with 'edge' and applying gaussian filter
                mean = np.pad(mean,
                              [(int(wsize / 2), int(wsize / 2)),
                               (int(wsize / 2), int(wsize / 2))],
                              'edge')
                std = np.pad(std, [(int(wsize / 2), int(wsize / 2)),
                                   (int(wsize / 2), int(wsize / 2))], 'edge')
            else:
                mean = np.zeros((len(vbl_levels), *vbl_dims))
                std = np.zeros_like(mean)
                for lon_idx in range(vbl_dims[1] - wsize + 1):
                    for lat_idx in range(vbl_dims[0] - wsize + 1):
                        mean[lat_idx, lon_idx] = np.mean(self.dataset[vbl_full_type][indices, :,
                                                         lat_idx:lat_idx + wsize, lon_idx:lon_idx + wsize],
                                                         axis=(0, -2, -1))
                        std[lat_idx, lon_idx] = np.mean(self.dataset[vbl_full_type][indices, :,
                                                        lat_idx:lat_idx + wsize, lon_idx:lon_idx + wsize],
                                                        axis=(0, -2, -1))

                # change dimension back py padding with 'edge' and applying gaussian filter
                mean = np.pad(mean,
                              [(0, 0), (int(wsize / 2), int(wsize / 2)),
                               (int(wsize / 2), int(wsize / 2))],
                              'edge')
                std = np.pad(std, [(0, 0), (int(wsize / 2), int(wsize / 2)),
                                   (int(wsize / 2), int(wsize / 2))], 'edge')

            mean = gaussian_filter1d(mean, sigma=10, axis=-1, mode='wrap')  # wrap around as it it is the longitude
            mean = gaussian_filter1d(mean, sigma=10, axis=-2, mode='nearest')
            std = gaussian_filter1d(std, sigma=10, axis=-1, mode='wrap')  # wrap around as it it is the longitude
            std = gaussian_filter1d(std, sigma=10, axis=-2, mode='nearest')

            return {"mean": mean,
                    "std": std}

        elif nt in ["pixelwise"]:
            mean = np.mean(self.dataset[vbl_full_type][indices], axis=0)
            std = np.std(self.dataset[vbl_full_type][indices], axis=0)
            return {"mean": mean,
                    "std": std}
        elif nt in ["framewise"]:
            mean = np.mean(self.dataset[vbl_full_type][indices])
            std = np.std(self.dataset[vbl_full_type][indices])
            return {"mean": mean,
                    "std": std}
        elif nt in ["logscale"]:
            return {}

    def __call__(self, sample):

        for b, sample_mode in enumerate(sample[0]["__sample_modes__"]):
            for section_k, section_v in sample[0].items():
                if section_k[:2] == "__":
                    continue
                for vbl_k, vbl_v in section_v.items():
                    if vbl_k[-4:] == "__ts":
                        continue
                    assert (sample_mode, section_k, vbl_k) in self.normalisation_cache, \
                        "Key {} not in normalisation cache! Maybe you should rebuild the cache?".format(
                            (sample_mode, section_k, vbl_k))
                    key = (sample_mode, section_k, vbl_k)
                    cache_dct = self.normalisation_cache[key]
                    sample[0][section_k][vbl_k][b] = th.from_numpy(
                        self.normalise(cache_dct, sample[0][section_k][vbl_k][b], sample[0]))

        return sample

    def normalise(self, cache, data, sample):
        normalisation_str = cache["normalisation_str"]

        nt = normalisation_str.split(":")
        if len(nt) > 1:
            nc = nt[1]
        else:
            nc = {}

        splt = nt[0].split("@")
        ac = None if len(splt) <= 1 else splt[1]
        nt = splt[0]

        if nt in ["lalas", "las", "framewise", "pixelwise", "global"]:
            if ac != "precip":
                data = data - cache["cache"]["mean"]
            res = np.divide(data,
                            cache["cache"]["std"],
                            out=np.zeros_like(data),
                            where=cache["cache"]["std"] != 0)  # NaNs are replaced by zeros!
            if ac == "precip":
                res = np.log(res)
                res[res != res] = -9
        elif nt in ["identity"]:
            res = data
        else:
            raise Exception("Unknow normalisation function: {}".format(nt))
        return res

    def _normalise_las(self, sample_vbl, sample, cache):

        pass

    def _normalise_pixelwise(self, sample_vbl, sample, cache):
        return (sample_vbl - cache["mean"]) / cache["std"]


if __name__ == "__main__":

    datapath = "/data/mmap/era5625/era5625.dill"
    partition_type = 'repeat'
    p_type = "repeat"

    if p_type == 'range':
        increments = 60 * 60
        times = {
            'train': (datetime(2010, 1, 1, 1, 0), datetime(2017, 1, 1)),
            'valid': (datetime(2017, 1, 15, 1, 0), datetime(2018, 12, 31, 23, 0))}
        train_seconds = (times['train'][1] - times['train'][0]).total_seconds()
        dump_seconds = (times['valid'][0] - times['train'][1]).total_seconds()
        val_seconds = (times['valid'][1] - times['valid'][0]).total_seconds()
        partition_conf = {
            "timerange": (times['train'][0].timestamp(), times['valid'][1].timestamp()),
            "partitions": [{"name": "train", "len_s": train_seconds, "increment_s": increments},
                           {"name": "dump", "len_s": dump_seconds, "increment_s": increments},
                           {"name": "valid", "len_s": val_seconds, "increment_s": increments}]
        }
    else:
        partition_conf = {"timerange": (datetime(2010, 1, 1, 0).timestamp(),
                                        datetime(2017, 1, 1, 0).timestamp()),
                          # Define partition elements
                          "partitions": [{"name": "train", "len_s": 20 * 24 * 60 * 60, "increment_s": 60 * 60},
                                         {"name": "val", "len_s": 10 * 24 * 60 * 60, "increment_s": 60 * 60},
                                         {"name": "test", "len_s": 10 * 24 * 60 * 60, "increment_s": 60 * 60}]}

    # partition_conf = {"timerange": (datetime.datetime(2017, 1, 1, 0).timestamp(),
    #                                 datetime.datetime(2017, 12, 31, 0).timestamp()),
    #                   # Define partition elements
    #                   "partitions": [{"name": "train", "len_s": 300 * 24 * 60 * 60, "increment_s": 60 * 60},
    #                                  {"name": "val", "len_s": 65 * 24 * 60 * 60, "increment_s": 60 * 60}]}
    # partition_type = "repeat"

    # partition_conf = [("train",  (datetime.datetime(2010,1,1,0).timestamp(), datetime.datetime(2017, 1,1,0).timestamp())),
    #                  ("test", (datetime.datetime(2017,1,1,0).timestamp(), datetime.datetime(2019, 1,1,0).timestamp()))]
    # partition_type = "range"
    history = np.array([0, -3, -6, -9, -12]) * 3600
    lead_times = np.arange(12, 36 + 12, 12) * 3600
    sample_conf = {"lead_time_{}".format(int(lt / 3600)):  # sample modes
        {
            "label":  # sample sections
                {
                    "lsm": {"vbl": "era5625/lsm"},
                    "lat2d": {"vbl": "era5625/lat2d"},
                    "t-600": {"vbl": "era5625/t:600hPa",
                              "t": history,
                              "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    "q-1000": {"vbl": "era5625/q:1000hPa",
                               "t": history,
                               "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "lsm": {"vbl": "era5625/lsm"},  # sample variables
                    # "lat": {"vbl": "era5625/lat2d"},
                    # "t": {"vbl": "era5625/t",
                    ##      "t": np.array([0, -1, -2, -3, ]) * 3600,
                    #      "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "q925": {"vbl": "era5625/q:925hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "q850": {"vbl": "era5625/q:850hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "q700": {"vbl": "era5625/q:700hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "q600": {"vbl": "era5625/q:600hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "q500": {"vbl": "era5625/q:500hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "q400": {"vbl": "era5625/q:400hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "vo925": {"vbl": "era5625/vo:925hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "vo850": {"vbl": "era5625/vo:850hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "vo700": {"vbl": "era5625/vo:700hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "vo600": {"vbl": "era5625/vo:600hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r1000": {"vbl": "era5625/r:1000hPa",
                    #           "t": history,
                    #           "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r925": {"vbl": "era5625/r:925hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r850": {"vbl": "era5625/r:850hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r700": {"vbl": "era5625/r:700hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r600": {"vbl": "era5625/r:600hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r500": {"vbl": "era5625/r:500hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "r-400": {"vbl": "era5625/r:400hPa",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "clbt": {"vbl": "simsat_lowres/clbt",
                    #          "t": history,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]}
                    # "tcc": {"vbl": "era5625/tcc",
                    #         "t": np.array([0]) * 3600,
                    #         "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    # "t1000": {"vbl": "era5625/t:1000hPa",
                    #          "t": np.array([0, -1, -2, -3, -4]) * 3600,
                    #          "interpolate": ["nan", "nearest_past", "nearest_future"][1]}

                },
            "target":
                {
                    "tp": {"vbl": "era5625/tp",
                           "t": np.array([lt]),
                           "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    "tcc": {"vbl": "era5625/tcc",
                            # "t": [1, 3, 6, 9],
                            "t": [lt],
                            "interpolate": ["nan", "nearest_past", "nearest_future"][1]}
                }
        }
        for lt in lead_times}  # np.array([1, 3, 6, 9]) * 3600}

    # Met-Net style: different targets per label -- as an option

    dataset = Dataset(datapath=datapath,
                      partition_conf=partition_conf,
                      partition_type=partition_type,
                      partition_selected="train",
                      sample_conf=sample_conf,
                      )

    print(len(dataset))

    val_dataset = Dataset(datapath=datapath,
                          partition_conf=partition_conf,
                          partition_type=partition_type,
                          partition_selected="val",
                          sample_conf=sample_conf,
                          )

    print(len(val_dataset))

    sample_conf = {"lead_time_{}".format(int(lt / 3600)):  # sample modes
        {
            "label":  # sample sections
                {
                    "lsm": {"vbl": "era5625/lsm"},
                    "lat2d": {"vbl": "era5625/lat2d"},
                    "t-600": {"vbl": "era5625/t:600hPa",
                              "t": history,
                              "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    "q-1000": {"vbl": "era5625/q:1000hPa",
                               "t": history,
                               "interpolate": ["nan", "nearest_past", "nearest_future"][1]}
                },
            "target":
                {
                    "tp": {"vbl": "era5625/tp",
                           "t": np.array([lt]),
                           "interpolate": ["nan", "nearest_past", "nearest_future"][1]},
                    "tcc": {"vbl": "era5625/tcc",
                            # "t": [1, 3, 6, 9],
                            "t": [lt],
                            "interpolate": ["nan", "nearest_past", "nearest_future"][1]}
                }
        }
        for lt in lead_times}  # np.array([1, 3, 6, 9]) * 3

    # Scenario 1b: PyTorch dataloader

    num_workers = [0, 8, 16, 32, 64]  # [16, 32, 64] #[0, 4, 16, 32, 64]
    batch_sizes = [32]  # [4, 32, 64, 128, 1024]

    res_dict = {}
    for numw in num_workers:
        for bs in batch_sizes:
            print("RUNNING: bs {} || num workers: {}".format(bs, numw))
            params = {'batch_size': bs,
                      'shuffle': True,
                      'num_workers': numw}

            dataloader = torch.utils.data.DataLoader(dataset, **params)
            import time

            t1 = time.time()
            for sample in tqdm(dataloader):
                import pdb;

                pdb.set_trace()
                pass
            t2 = time.time()
            print("Total time: {}s; per sample: {}s; samples/sec: {}".format(t2 - t1,
                                                                             (t2 - t1) / (len(dataloader) * bs),
                                                                             float(len(dataloader)) * bs / (t2 - t1)))
            res_dict[(bs, numw)] = float(len(dataloader)) * bs / (t2 - t1)


    print(res_dict)

    # open a file, where you ant to store the data
    file = open('res_dict.pkl', 'wb')

    import pickle

    # dump information to that file
    pickle.dump(res_dict, file)

    pass
