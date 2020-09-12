import os
import pandas as pd
import numpy as np
import json

from matplotlib import pyplot as plt
plt.style.use('ggplot')

# make plots
with open("./results/era5625.json", "r") as f:
    era5625 = json.load(f)

with open("./results/era140625.json", "r") as f:
    era140625 = json.load(f)

with open("./results/imerg5625.json", "r") as f:
    imerg5625 = json.load(f)

with open("./results/imerg140625.json", "r") as f:
    imerg140625 = json.load(f)

with open("./results/imerg_25bi.json", "r") as f:
    imerg25 = json.load(f)
markers = ('+', '1', '+', '1', 'x')

# create plot
tcks = era5625["hist_den"][1]

fig = plt.figure(figsize=(8, 3))
plt.ylabel('probability density')
plt.xlabel('precipitation [mm/hour]')
ax = fig.gca()
ax.set_yscale("log")
ax.axvline(0, linestyle="-", color="white")
ax.axvline(2, linestyle="--", color="white")
ax.axvline(10, linestyle="--", color="white")
ax.axvline(50, linestyle="--", color="white")
ax.plot(tcks[:-1], era140625["hist_den"][0], label="ERA:tp $1.40625^\circ$", linestyle='None', marker=markers[1])
ax.plot(tcks[:-1], era5625["hist_den"][0], label="ERA:tp $5.625^\circ$", linestyle='None', marker=markers[0]) 
ax.plot(tcks[:-1], imerg25["hist_den"][0], label="IMERG $0.25^\circ$", linestyle='None', marker=markers[4])
ax.plot(tcks[:-1], imerg140625["hist_den"][0], label="IMERG $1.40625^\circ$", linestyle='None', marker=markers[3])
ax.plot(tcks[:-1], imerg5625["hist_den"][0], label="IMERG $5.625^\circ$", linestyle='None', marker=markers[2])
ax.legend()
plt.grid()
fig.savefig("hist_den.pdf", bbox_inches='tight')

