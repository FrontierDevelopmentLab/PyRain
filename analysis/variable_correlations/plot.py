import os
import pandas as pd
import numpy as np
import json

from matplotlib import pyplot as plt

# open results files
with open("./out.json") as f:
    res = json.load(f)

d500 = np.array(res["300"]["5625"])
d500h = np.array(res["850"]["5625"])


t_tot = d500 + d500h.transpose() + np.eye(d500.shape[0])

t_tot[t_tot!=t_tot] = 0.0

import seaborn as sns
corr = t_tot
tick_labels = ["longitude (lon)",
               "latitude (lat)",
               "land-sea mask (lsm)",
               "orography (oro)",
               "soil type (slt)",
               "geopotential height (z)",
               "temperature (t)",
               "specific humidity (q)",
               "surface pressure (sp)",
               "cloud liquid water content (clwc)",
               "cloud ice water content (ciwc)",
               "temperature at 2m (t2m)",
               "SimSat channel 0 (clbt:0)",
               "SimSat channel 1 (clbt:1)",
               "SimSat channel 2 (clbt:2)",
               "ERA5 total precipitation (tp)",
               "IMERG precipitation"]
tick_labels_short = ["lon", "lat", "lsm", "oro", "slt",
                     "z", "t", "q", "sp", "clwc", "ciwc",
                     "t2m", "clbt:0", "clbt:1", "clbt:2",
                     "tp", "imerg"]
sns.heatmap(t_tot,
            xticklabels=tick_labels_short,
            yticklabels=tick_labels_short,
            cmap= 'coolwarm', vmin=-1, vmax=1,
            annot = False,
            center=0)

plt.savefig("corr_matrix.pdf", bbox_inches='tight')


