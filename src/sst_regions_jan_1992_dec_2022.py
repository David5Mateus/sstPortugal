import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cycler import cycler


coordinates_file_path = '../data/sites_coordinates.csv'
retrieved_dataset = '../data/temp_last version2.nc'

output_folder = "../outputs"
output_filename = "sst_regions_1992_2022_mean_plot.pdf"
output_filepath = f"{output_folder}/{output_filename}"

regions = ['North', 'Center', 'Southwest', 'South']
line_colors = ['blue', 'orange', 'green', 'red']

custom_cycler = cycler(color=line_colors)


def build_sst_mean_plot():
    ds = xr.open_dataset(retrieved_dataset)
    # convert to celsius
    sst = ds.analysed_sst - 273.15
    # calculate yearly mean values
    sst_yearly = sst.resample(time='Y').mean()

    # load coordinates file dataset
    stns = pd.read_csv(coordinates_file_path, sep=';')

    # Matriz bidimensional para guardar os valores
    sst2d = np.empty((len(stns), len(sst_yearly))) * np.nan

    # Preenchimento da matriz
    for idx in range(0, len(stns)):
        stn_lo = stns.Longitude[idx]
        stn_la = stns.Latitude[idx]

        sst2d[idx, :] = sst_yearly.sel(longitude=stn_lo, latitude=stn_la, method='nearest')

    sst_loc = xr.Dataset(
        data_vars=dict(
            sst=(["loc", "time"], sst2d),
            region=(["loc"], stns.Regiao.values),
            local=(["loc"], stns.Local.values),
            lat=(["loc"], stns.Latitude.values),
            lon=(["loc"], stns.Longitude.values),
        ),
        coords=dict(
            loc=(["loc"], stns.index.array),
            time=sst_yearly.time,
        ),
        attrs=dict(description=""),
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    # PLOT SETTINGS
    # Line colors
    ax.set_prop_cycle(custom_cycler)
    # Set X and Y labels
    ax.set_xlabel('')
    ax.set_ylabel('°C (mean ± S.E.)')
    # Format time on x-axis
    date_format = mdates.DateFormatter('%b %Y')  # Month and Year
    ax.xaxis.set_major_formatter(date_format)
    # Set the interval of time on the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=36))

    for region, color in zip(regions, line_colors):
        sst_data = sst_loc.where(sst_loc.region == region).sst

        # Plot mean line with specified color
        sst_mean = sst_data.mean('loc')
        ax.plot(sst_mean.time.values, sst_mean.values, label=f'{region} (mean)', color=color)

        # Calculate standard error using numpy's std function and divide by sqrt(number of samples)
        sst_se = sst_data.std('loc', ddof=1) / np.sqrt(len(stns))

        # Plot standard error bars without points
        # ax.errorbar(
        #     sst_mean.time.values, sst_mean.values, yerr=sst_se, fmt='-', label='', capsize=5, elinewidth=1, ecolor='black')

    # Create a legend with only region names
    ax.legend(regions, title='Regions')

    return fig


def export_sst_mean_plot(fig):
    # Exports the plot to a PDF file
    # Save the figure to the specified PDF file
    fig.savefig(output_filepath, bbox_inches="tight", format="pdf")


if __name__ == '__main__':
    fig = build_sst_mean_plot()
    # plt.show()
    export_sst_mean_plot(fig)
