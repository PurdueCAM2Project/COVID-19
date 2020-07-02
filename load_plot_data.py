import pandas as pd 
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import seaborn as sns

cars = "/content/cars.csv"
people = "ppl.csv"

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
    df2 = df.groupby("date").agg(cols).reset_index()

    length = None
    date = dt.strptime("2020-03-30", "%Y-%m-%d")
    for key in values:
        mask = df["cam_id"].apply(lambda x: x == key)
        data, max_date, start = get_count(df, mask,key = col)
        data = data.set_index("date")
        if max_date > date:
            date = max_date
        length = sum(mask.apply(lambda x: 1 if x else 0))
        df_masks.update({key: {"mask":mask, "length":length, "data": data, "last_date": max_date, "first_date": start}})
    
    start = dt.strptime("2020-03-30", "%Y-%m-%d")
    td = delta(days = 1)
    dates = [start]
    while start < date:
        start += td
        dates.append(start)
    dates = pd.DataFrame(dates, columns=['date_keys'])
    return df_masks, dates

def construct_new(data, dates, col = "pedestrian_count"):
    frames = []
    for key in data.keys():
        data[key]["data"] = data[key]["data"].reindex(dates["date_keys"])
        data[key]["data"].columns = [key]
        frames.append(data[key]["data"])

    datafin = pd.concat(frames, axis = 1)
    return data , datafin

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
    
    keys, dates = ungroup(data, keys, col = col)
    keys, flattened = construct_new(keys, dates)

    flattened = flattened.sort_index()
    flattened = flattened.reset_index()
    flattened["date_keys"] = flattened["date_keys"].apply(lambda x: str(x).replace("00:00:00", "").replace("2020-", ""))

    print(f" num cams: {len(keys)}")
    print(f" num cams check: {len(flattened.columns)}")
    return data, keys, dates, flattened

def plot(frame, index_col = "date_keys", plot_list = None, fill_na_value = None, height = 10, aspect = 2, kind = 'scatter', alpha=0.5):

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
    plot_frame = pd.melt(plot_frame, 'date_keys', var_name='IDs', value_name=f"{col}")
    g = sns.relplot(x="date_keys", y=f"{col}", hue = "IDs", height=height, aspect=aspect, data=plot_frame, alpha=alpha, kind=kind)
    g.set_xticklabels(rotation=90)
    g.set_titles("cams vs dates")
    return g

if __name__ == "__main__":
    col = "pedestrian_count"
    data, keys, dates, colap = load_csv(people, col)
    #plot_cams = ["jqaxhvDafz/"]
    plot_cams = 'all'
    g = plot(colap, plot_list = plot_cams, fill_na_value = None, alpha = 0.9, aspect = 2, height = 7)
    plt.show()