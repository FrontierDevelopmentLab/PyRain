from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import iris
import iris.plot as iplt
import cartopy.crs as ccrs
from datetime import datetime
from run_benchmark import RegressionModel


def collect_predictions(model_path, date):
    """
    Given model path, load learnt model to predict for all lead times, using an input sequence from a specific date.
    Output has shapes (no. of lead times, lat, lon)
    """
    # Collect data to plot
    model, hparams, loaderDict, normalizer, collate = RegressionModel.load_model(model_path)
    samples= []
    lead_times = hparams['lead_times']
    for i in lead_times:
        samplet = f'lead_time_{i}'
        sample_lt = loaderDict['test'].get_sample_at(samplet, date.timestamp())
        sample_lt['__sample_modes__'] = samplet
        samples.append([sample_lt])
    # make predictions
    sample_x, truth, lts = collate(samples)
    model.eval()
    out = model(sample_x).detach().numpy()
    truth = truth.numpy()
    return out[:, 0, :, :], truth[:, 0, :, :], hparams


def determine_bounds(img: list):
    """
    determine min and max values found in samples
    """
    vmin = min([im.min() for im in img])
    vmax = max([im.max() for im in img])
    return vmin, vmax


def make_use_of_cube_data(nc_path: str):
    """
    load any nc file (e.g. 'total_precipitation_2019_5.625deg.nc') in order to make use of its structure for plotting later.
    """
    cube = iris.load_cube(nc_path)
    cube = cube[0,:,:]# Remove time             
    cube.coord('lat').rename('latitude')
    cube.coord('lon').rename('longitude')
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    return cube


def plot_one_img(cube, inarray, vmin, vmax, path=None):
    """
    Given the cube structure, the data array, together with min/ max values for scaling, output a projected plot over earth.
    """
    # replace data in cube with incoming array.
    cube.data = inarray[:,:]

    fig=plt.figure(dpi=400,figsize=(8,4))
    # determine colour bounds
    delta = (vmax - vmin) / 100
    bounds = np.arange(vmin, vmax+delta,delta)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = plt.get_cmap('viridis') 
    # project image
    projection=ccrs.Mollweide
    ax = plt.axes(projection=projection(central_longitude=0))
    iplt.pcolormesh(cube ,norm=norm,cmap=cmap)                                                                                                              
    ax.coastlines()
    # transparent background
    ax.patch.set_facecolor('#ababab')
    ax.patch.set_alpha(0)
    fig.clf()
    if path is not None:
        fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')


def main(hparams):
    """
    Main function for plotting truth and predictions.
    """
    date = datetime(2019, 7, 12) # random
    load_path = hparams.load
    nc_file = hparams.nc_file
    hparams = vars(hparams)
    out, truth, hparams = collect_predictions(load_path, date)
    cube = make_use_of_cube_data(nc_file)
    vmin, vmax = determine_bounds([out, truth])
    for t, _ in enumerate(hparams['lead_times']):
        # plot predictions
        plot_one_img(cube, out[t], vmin, vmax)
        # plot ground truth
        plot_one_img(cube, truth[t], vmin, vmax)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load", required=True, type=str, help='Path of checkpoint directory to load')
    parser.add_argument("--nc_file", required=True, type=str, help='Path of a random .nc file')
    hparams = parser.parse_args()
    main(hparams)