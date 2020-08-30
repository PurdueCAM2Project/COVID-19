"""
This file provides tools to convert a raw dataframe of detections to a dataframe aligned by date, for ease of plotting.
There is also functionality to plot by city, state, plot vehicles or people.

"""
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

def get_count(df, mask, key):

    cols = {key: np.max}
    data = df[mask]
    df = df.groupby("date").agg(cols).reset_index()
    red = data.groupby("date").agg(cols).reset_index()
    red["date"] = red["date"].apply(lambda x: dt.strptime(x, "%Y-%m-%d"))
    return red, max(red["date"]), dt.strptime("2020-03-30", "%Y-%m-%d")


def ungroup(df, values, col):
    df_masks = {}
    cols = {col: np.max}
    # df2 = df.groupby("date").agg(cols).reset_index()
    cam_ids = np.array(df["cam_id"])

    length = None
    date = dt.strptime("2020-03-30", "%Y-%m-%d")
    for key in tqdm(values):
        mask = cam_ids == key
        data, max_date, start = get_count(df, mask, key=col)
        data = data.set_index("date")
        if max_date > date:
            date = max_date
        length = np.sum(mask)
        df_masks.update({key: {"mask": mask, "length": length,
                               "data": data, "last_date": max_date, "first_date": start}})

    start = dt.strptime("2020-03-30", "%Y-%m-%d")
    td = delta(days=1)
    dates = [start]
    while start < date:
        start += td
        dates.append(start)
    dates = pd.DataFrame(dates, columns=['date_keys'])
    return df_masks, dates


def construct_new(data, dates, col="pedestrian_count"):
    frames = []
    for key in data.keys():
        data[key]["data"] = data[key]["data"].reindex(dates["date_keys"])
        data[key]["data"].columns = [key]
        frames.append(data[key]["data"])

    datafin = pd.concat(frames, axis=1)
    return data, datafin


def load_csv(filen, col):
    """
    load the csv and flatten all ids into individual columns

    Args:
        filen: string path to file
        col: the column that you would want to plot
                [pedestrian_count, vehicle_count, ...]

    return:
        data: pandas dataframe, unaltered data
        keys: dictionary of the cam_id and the data associate with that cam id
                structure {key: {"mask":mask, "length":length, "data": data, "last_date": max_date, "first_date": start}}
        dates: the list of dates to plot against on the x axis, un altered
        flattened: the data frame with each camera haveing its own column, the data in the column in the data specified in
                   input param 'col', all the data points re indexed to fit the same x axis
    """
    data = pd.read_csv(filen)
    data = data.fillna(0)
    keys = set(list(data["cam_id"].values))

    with open('keys_cars.txt', 'w') as file:
        print(len(keys))
        for each in keys:
            if len(each)>15:
                file.write(each + '\n')
                print(each)

    keys, dates = ungroup(data, keys, col=col)
    keys, flattened = construct_new(keys, dates)

    flattened = flattened.sort_index()
    flattened = flattened.reset_index()
    flattened["date_keys"] = flattened["date_keys"].apply(
        lambda x: str(x).replace("00:00:00", "").replace("2020-", ""))

    print(f" num cams: {len(keys)}")
    print(f" num cams check: {len(flattened.columns)}")
    return data, keys, dates, flattened


def plot(frame, index_col="date_keys", plot_list=None, fill_na_value=None, height=10, aspect=2, kind='scatter', alpha=0.5):
    '''
    quick plotting function to plot the data frame as a line or scatter plot if data needs to plotted the same way as example

    Args:
        frame: the data frame the plot
        index_col: the column to use as x axis
        plot_list: key word 'all' to plot data for all camera IDs
                   a string with the camera id you want to plot, format: <insert id>/ -> ex. jqaxhvDafz/
                   a list of the cameras you would like to plot, follow format above for all cams in list
        fill_na_value: float value, by default set to -np.inf, (-np.inf) tell seaborn to ignore data
        height: float for the height of graph in centimeters, default is 10
        aspect: float for aspect ratio to follow, default is 2
        kind: string, [line, scatter], default is scatter
        alpha: float btwn 0, 1 for the opacity of the graphs

    Return:
        g: matplotlib plot object

    Caveats:
        the function plots the curve, it is your job to call plt.show() after this function
    '''
    sns.set_style("darkgrid")
    if plot_list == 'all':
        plot_list = list(frame.columns[1:])
    elif type(plot_list) == str:
        plot_list = [plot_list]
    else:
        plot_list = list(plot_list)
    index_col = [index_col] + plot_list

    if fill_na_value != None:
        frame = frame.fillna(fill_na_value)
    else:
        frame = frame.fillna(-np.inf)

    plot_frame = frame.loc[:, index_col]
    plot_frame = pd.melt(plot_frame, 'date_keys',
                         var_name='IDs', value_name=f"{col}")
    g = sns.relplot(x="date_keys", y=f"{col}", hue="IDs", height=height,
                    aspect=aspect, data=plot_frame, alpha=alpha, kind=kind)
    g.set_xticklabels(rotation=90)
    g.set_titles("cams vs dates")
    return g


def get_plot_cams_list(people_df, city=None, country=None, state=None):
    if country:
        return list(set(people_df.loc[people_df['country'] == country, 'cam_id']))
    elif city:
        return list(set(people_df.loc[people_df['city'] == city, 'cam_id']))
    elif state:
        return list(set(people_df.loc[people_df['state'] == state, 'cam_id']))


def get_max_of_subset(colap, list_cams):
    return list(colap.max(colap[list_cams]))

def visualize(ax, place_to_use, short_form, useful_dates, colap_vehicle=None, colap_people=None, data_vehicles=None, data_people=None,  disclude_vehicles=None, disclude_people=None):

    if colap_vehicle is not None:
        # chuck the data of march 30, unsure if accurate
        useful_data = colap_vehicle[1:]
        n_weeks = len(useful_dates)
        data_points = len(useful_data)

        start_samples = []
        end_samples = []
        plot_dates = []

        for i in range(n_weeks):
            start_samples.append(7 * i)
            end_samples.append(min(7 * (i + 1), data_points))

        for i in range(n_weeks):
            # uncomment the first one if you want dates of the form startdate-enddate
            # The uncommented part now makes dates of the form startdate

            # plot_dates.append(np.array(useful_data[start_samples[i]:start_samples[i]+1]["date_keys"])[
            #                   0]+"to "+np.array(useful_data[end_samples[i]-1:end_samples[i]]["date_keys"])[0][:-1])
            plot_dates.append(np.array(useful_data[start_samples[i] + 1:start_samples[i] + 2]["date_keys"])[0])

        weekly_counts = np.zeros(len(plot_dates))
        plot_cams = get_plot_cams_list(data_vehicles, country=short_form)

        for i in range(n_weeks):
            data_to_use = useful_data[start_samples[i]:end_samples[i]][plot_cams]
            weekly_counts[i] = np.max(np.sum(data_to_use[plot_cams], axis=1))

    if colap_people is not None:
        weekly_counts_people = np.zeros(len(plot_dates))
        plot_cams = get_plot_cams_list(data_people, country=short_form)


        useful_data_people=colap_people[1:]
        for i in range(n_weeks):
            data_to_use = useful_data_people[start_samples[i]:end_samples[i]][plot_cams]
            weekly_counts_people[i] = np.max(np.sum(data_to_use[plot_cams], axis=1))

    ax.set(title=place_to_use)

    if colap_vehicle is not None:
        if disclude_vehicles:
            if short_form not in disclude_vehicles:
                ax.bar(plot_dates, weekly_counts, align='edge', width=0.4, color='white', edgecolor='black')
                ax.set_ylabel('Number of Vehicles')
        else:
            ax.bar(plot_dates, weekly_counts, align='edge', width=0.4, color='white', edgecolor='black')
            ax.set_ylabel('Number of Vehicles')

    if colap_people is not None:
        if disclude_people:
            if short_form not in disclude_people:
                ax3 = ax.twinx()
                ax3.bar(plot_dates, weekly_counts_people, align='edge', color='black', width=-0.4)
                ax3.set_ylabel('Number of People')
        else:
            ax3 = ax.twinx()
            ax3.bar(plot_dates, weekly_counts_people, align='edge', color='black', width=-0.4)
            ax3.set_ylabel('Number of People')

    ax.set_xticklabels(plot_dates, rotation=60)

    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)


if __name__ == "__main__":

    # Required preliminary steps

    col = "vehicle_count"
    cars = "combined_csv_vehicles.csv"
    colap_vehicle = pd.read_csv('processed_vehicles.csv')
    data = pd.read_csv(cars)
    data_vehicles = data.fillna(0)

    col = "pedestrian_count"
    people = "combined_csv_pedestrians_without_night_place.csv"
    colap_people = pd.read_csv('processed_people.csv')
    data = pd.read_csv(people)
    data_people = data.fillna(0)

    # Example: generating plots from paper

    fig, axes = plt.subplots(nrows=3, ncols=5)

    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Vehicle and People Count Trends in Select Countries')

    places_to_use = ["United States", "Australia", "France",
                     "Austria", "Denmark", "Great Britain",
                     "Czech Republic", "Switzerland", "Italy",
                     "Germany", "Canada", "New Zealand",
                     "Hong Kong", "Spain", "Hungary"]

    short_forms = ["USA", "AU", "FR",
                   "AT", "DK", "GB",
                   "CZ", "CH", "IT",
                   "DE", "CA", "NZ",
                   "HK", "ES", "HR"]

    useful_dates = [("April", 6), ("April", 13), ("April", 20), ("April", 27), ("May", 4),
                    ("May", 11), ("May", 18), ("May", 25), ("June", 1), ("June", 8), ("June", 15),
                    ("June", 22), ("June", 29), ("July", 6), ("July", 13), ("July", 20), ("July", 27), ("August", 1)]

    Dates = ["April 6", "April 13", "April 20", "April 27", "May 4",
             "May 11", "May 18", "May 25", "June 1", "June 8", "June 15", "June 22", "June 29", "July 6", "July 13",
             "July 20", "July 27", "August 1"]

    disclude_people = ["CA", "DE", "NZ", "HK", "ES", "HR"]
    disclude_vehicles = ["IT", "CH"]

    for ax, place_to_use, short_form in zip(axes.flatten(), places_to_use, short_forms):
        visualize(ax,  place_to_use, short_form, useful_dates, colap_vehicle, colap_people, data_vehicles, data_people, disclude_vehicles, disclude_people)

    plt.show()

    # Another example: Plotting only Vehicles in United States
    # place_to_use = "United States"
    # short_form = "USA"
    #
    # useful_dates = [("April", 6), ("April", 13), ("April", 20), ("April", 27), ("May", 4),
    #                                  ("May", 11), ("May", 18), ("May", 25), ("June", 1), ("June", 8), ("June", 15),
    #                                  ("June", 22), ("June", 29), ("July", 6), ("July", 13), ("July", 20), ("July", 27), ("August", 1)]
    # fig, ax1 = plt.subplots()
    # visualize(ax1, place_to_use, short_form, useful_dates, colap_vehicle=colap_vehicle, data_vehicles=data_vehicles)
    # plt.show()
