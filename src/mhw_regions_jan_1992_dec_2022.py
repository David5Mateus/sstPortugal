import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cycler import cycler


coordinates_file_path = '../data/sites_coordinates.csv'
retrieved_dataset = '../data/temp_last version2.nc'

output_folder = "../outputs"
output_filename = "mhw_regions_1992_2022_plot_2.pdf"
output_excel_filename = "mhw_regions_1992_2022.xlsx"
output_filepath = f"{output_folder}/{output_filename}"
output_excel_filepath = f"{output_folder}/{output_excel_filename}"

regions = ['North', 'Center', 'Southwest', 'South']
line_colors = ['blue', 'orange', 'green', 'red']

custom_cycler = cycler(color=line_colors)


def find_consecutive_days(df):
    df = df.sort_values('time')
    df['diff'] = df['time'].diff().dt.days
    df['consecutive'] = df['diff'] == 1

    # Find the start and end of consecutive sequences
    df['block'] = (~df['consecutive']).cumsum()
    blocks = df.groupby('block').filter(lambda x: len(x) >= 5)

    return blocks.groupby('block').agg(
        {'time': ['min', 'max', 'count']}).reset_index()


def build_sst_mean_plot():
    ds = xr.open_dataset(retrieved_dataset)
    # convert to celsius
    sst = ds.analysed_sst - 272.15
    # calculate daily mean values
    sst_daily = sst.resample(time='D').mean()

    # get 90th percentile
    q90 = sst_daily.quantile(0.9, dim="time")

    # load coordinates file dataset
    stns = pd.read_csv(coordinates_file_path, sep=';')

    # Matriz bidimensional para guardar os valores
    sst2d = np.empty((len(stns), len(sst_daily))) * np.nan

    # Preenchimento da matriz
    for idx in range(0, len(stns)):
        stn_lo = stns.Longitude[idx]
        stn_la = stns.Latitude[idx]

        sst2d[idx, :] = sst_daily.sel(
            longitude=stn_lo, latitude=stn_la, method='nearest')

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
            time=sst_daily.time,
        ),
        attrs=dict(description=""),
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    # PLOT SETTINGS
    # Line colors
    ax.set_prop_cycle(custom_cycler)
    # Set X and Y labels
    ax.set_xlabel('')
    ax.set_ylabel('Marine Heat Waves')
    # Format time on x-axis
    date_format = mdates.DateFormatter('%b %Y')  # Month and Year
    ax.xaxis.set_major_formatter(date_format)
    # Set the interval of time on the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=36))

    for region, color in zip(regions, line_colors):
        sst_data = sst_loc.where(sst_loc.region == region).sst
        # split data per year
        sst_yearly = sst_data.resample(time='YE')

        yearly_heat_waves = []

        # get min q90 for all years, for that region
        q90_min = float(sst_yearly.quantile(0.9, dim="time").min())

        # TODO: try doing mean for each site ['loc']

        for year in sst_yearly:
            # filter days where sst is above q90_min
            year_sst_above_q90 = year[1].where(year[1] > q90_min)
            df_sst_above_q90 = year_sst_above_q90.to_dataframe(
                name='temperature').reset_index().dropna()

            # group by location to find dates where sst is above q90 for 5
            # consecutive days or more and count date groups
            consecutive_days = df_sst_above_q90.groupby('loc').apply(
                find_consecutive_days).reset_index(level=0, drop=True)
            yearly_heat_waves.append(len(consecutive_days) / df_sst_above_q90['loc'].nunique())

        consecutive_days_da = xr.DataArray(
            yearly_heat_waves,
            coords=[sst_yearly.mean().time.values],
            dims=["year"],
            name="consecutive_days_count"
        )

        # Plot mean line with specified color
        ax.plot(sst_yearly.mean().time, consecutive_days_da, label=f'{region}', color=color)

    # Create a legend with only region names
    ax.legend(regions, title='Regions')

    return fig, sst_yearly.mean().time, consecutive_days_da


def export_sst_mean_plot(fig):
    # Exports the plot to a PDF file
    # Save the figure to the specified PDF file
    fig.savefig(output_filepath, bbox_inches="tight", format="pdf")


def export_csv(indexes, data):
    # FIXME: correct data, indexes and columns
    df = pd.DataFrame(data, index=regions, columns=indexes)
    df.to_excel(output_excel_filepath, sheet_name='Sheet 1')


if __name__ == '__main__':
    fig, indexes, data = build_sst_mean_plot()
    # plt.show()
    export_sst_mean_plot(fig)
    # export_csv(indexes, data)
