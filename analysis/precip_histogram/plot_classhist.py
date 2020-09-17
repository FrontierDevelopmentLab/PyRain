import pickle
from matplotlib import pyplot as plt
import numpy as np
import skimage.measure
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime


with open("/home/cs/Desktop/fracres.pkl", "rb") as f:
    res = pickle.load(f)

print("Pickle loaded!")

import seaborn as sns

f1_red = skimage.measure.block_reduce(res["f1"], (8,8), np.max)
fig, ax = plt.subplots(figsize=(10, 20))
im = ax.imshow(f1_red.transpose(), cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0.95, vmax=1.0, extent=[0, 360,0,180])

ax.set_xticks(np.linspace(0, 360, 11))
lonw = ["${:d}^\circ$W".format(int(c)) for c in reversed(np.linspace(36, 180, 5))]
lone = ["${:d}^\circ$E".format(int(c)) for c in np.linspace(36, 180, 5)]
lones = lonw + [0] + lone
ax.set_yticks(np.linspace(0, 180, 11))
latn = ["${:d}^\circ$N".format(int(c)) for c in reversed(np.linspace(18, 90, 5))]
late = ["${:d}^\circ$S".format(int(c)) for c in np.linspace(18, 90, 5)]
lates = latn + [0] + late
ax.set_xticklabels(lones)
ax.set_yticklabels(lates)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

ls = np.linspace(0.95, 1.0, 5)
cbar = fig.colorbar(im, cax=cax, ticks=ls)
cbar.set_ticklabels(["{:d}%".format(int(c*100.0)) for c in ls])
print("saving ...")
plt.savefig("f1.pdf", bbox_inches='tight')

#########################
f2_red = skimage.measure.block_reduce(res["f2"], (8,8), np.max)
fig, ax = plt.subplots(figsize=(10, 20))
im = ax.imshow(f2_red.transpose(), cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=0.05, extent=[0, 360,0,180])

ax.set_xticks(np.linspace(0, 360, 11))
lonw = ["${:d}^\circ$W".format(int(c)) for c in reversed(np.linspace(36, 180, 5))]
lone = ["${:d}^\circ$E".format(int(c)) for c in np.linspace(36, 180, 5)]
lones = lonw + [0] + lone
ax.set_yticks(np.linspace(0, 180, 11))
latn = ["${:d}^\circ$N".format(int(c)) for c in reversed(np.linspace(18, 90, 5))]
late = ["${:d}^\circ$S".format(int(c)) for c in np.linspace(18, 90, 5)]
lates = latn + [0] + late
ax.set_xticklabels(lones)
ax.set_yticklabels(lates)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

ls = np.linspace(0, 0.05, 5)
cbar = fig.colorbar(im, cax=cax, ticks=ls)
cbar.set_ticklabels(["{:d}%".format(int(c*100.0)) for c in ls])
print("saving ...")
#plt.show()
plt.savefig("f2.pdf", bbox_inches='tight')


#########################
f3_red = skimage.measure.block_reduce(res["f3"], (8,8), np.max)
fig, ax = plt.subplots(figsize=(10, 20))
im = ax.imshow(f3_red.transpose(), cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=0.01, extent=[0, 360,0,180])

ax.set_xticks(np.linspace(0, 360, 11))
lonw = ["${:d}^\circ$W".format(int(c)) for c in reversed(np.linspace(36, 180, 5))]
lone = ["${:d}^\circ$E".format(int(c)) for c in np.linspace(36, 180, 5)]
lones = lonw + [0] + lone
ax.set_yticks(np.linspace(0, 180, 11))
latn = ["${:d}^\circ$N".format(int(c)) for c in reversed(np.linspace(18, 90, 5))]
late = ["${:d}^\circ$S".format(int(c)) for c in np.linspace(18, 90, 5)]
lates = latn + [0] + late
ax.set_xticklabels(lones)
ax.set_yticklabels(lates)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

ls = np.linspace(0, 0.01, 5)
cbar = fig.colorbar(im, cax=cax, ticks=ls)
cbar.set_ticklabels(["{:d}%".format(int(c*100.0)) for c in ls])
print("saving ...")
#plt.show()
plt.savefig("f3.pdf", bbox_inches='tight')

#########################
f4_red = skimage.measure.block_reduce(res["f4"], (8,8), np.max)
fig, ax = plt.subplots(figsize=(10, 20))
im = ax.imshow(f4_red.transpose(), cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=0.001, extent=[0, 360,0,180])

ax.set_xticks(np.linspace(0, 360, 11))
lonw = ["${:d}^\circ$W".format(int(c)) for c in reversed(np.linspace(36, 180, 5))]
lone = ["${:d}^\circ$E".format(int(c)) for c in np.linspace(36, 180, 5)]
lones = lonw + [0] + lone
ax.set_yticks(np.linspace(0, 180, 11))
latn = ["${:d}^\circ$S".format(int(c)) for c in reversed(np.linspace(18, 90, 5))]
late = ["${:d}^\circ$N".format(int(c)) for c in np.linspace(18, 90, 5)]
lates = latn + [0] + late
ax.set_xticklabels(lones)
ax.set_yticklabels(lates)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

ls = np.linspace(0, 0.001, 5)
cbar = fig.colorbar(im, cax=cax, ticks=ls) 
cbar.set_ticklabels(["{:.2f}%".format(c*100.0) for c in ls])
print("saving ...")
plt.savefig("f4.pdf", bbox_inches='tight')
