""" This file plots scatterplots of a specified country or US state with optional color
    coding for date ranges.
"""


from plotting_tools import *
from matplotlib.patches import Patch
import pycountry
from matplotlib.lines import Line2D
from us_states import us_state_abbrev
import argparse

def color_list(plot_dates, date1=None, date2=None, date3=None, date4=None):
    if (date1):
        point1 = plot_dates.index(date1)
    if (date2):
        point2 = plot_dates.index(date2)
    if (date3):
        point3 = plot_dates.index(date3)
    if (date4):
        point4 = plot_dates.index(date4)
    
    if date4:
        return ["r"]*(point1) + ["g"]*(point2 - point1) + ["b"]*(point3-point2) + ["black"]*(point4-point3) + ["darkorchid"]*(len(plot_dates)-point4),  ['x']*(point1)+['o']*((point2) - point1) + ['*']*(point3 - point2) + ['+']*(point4 - point3) + ['D']*(len(plot_dates)-point4)

    elif date2:
        return ["r"]*(point1) + ["g"]*(((point2)) - point1) + ["blue"]*(len(plot_dates)-point2),  ['x']*(point1)+['o']*(point2 - point1) + ['*']*(len(plot_dates) - point2)

    elif date1:
        return ["r"]*(point1) + ["g"]*(len(plot_dates) - point1),  ['x']*(point1)+['o']*(len(plot_dates) - point1)

    else:
        return ["r"]*len(plot_dates), ['x']*len(plot_dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scatterplots by location.')
    parser.add_argument('--country', type=str,
                        help='country to plot')
    parser.add_argument('--state', type=str,
                        help=' US state to plot')
    parser.add_argument('--date1', type=str,
                        help='first date of interest', required=False, default=None)
    parser.add_argument('--date2', type=str,
                        help='second date of interest', required=False, default=None)
    parser.add_argument('--date3', type=str,
                        help='third date of interest', required=False, default=None)
    parser.add_argument('--date4', type=str,
                        help='fourth date of interest', required=False, default=None)

    args = parser.parse_args()

    date1 = args.date1
    date2 = args.date2
    date3 = args.date3
    date4 = args.date4
    country = args.country
    state = args.state

    fig, ax = plt.subplots()


    countries = {}
    for countryname in pycountry.countries:
        countries[countryname.name] = countryname.alpha_2

    if country:
        short_form = countries.get(country, 'Unknown code')
        place_to_use = country
    elif state:
        short_form = us_state_abbrev[state]
        place_to_use = state


    """vehicle counts ----------------------------------------------------------
    """

    col = "vehicle_count"
    cars = "combined_csv_vehicles_4-1_8-1.csv"
    colap = pd.read_csv('processed_vehicles.csv')
    data = pd.read_csv(cars)
    data = data.fillna(0)

    # remove the data of March 31
    useful_data = colap[1:]
    data_points = len(useful_data)

    start_samples = []
    end_samples = []
    plot_dates = []

    for i in range(data_points):
        start_samples.append(1*i)
        end_samples.append(min(1*(i+1), data_points))

    for i in range(data_points):
        plot_dates.append(np.array(useful_data[start_samples[i]:start_samples[i]+1]["date_keys"])[0])

    daily_counts = np.zeros((data_points))

    if country:
        plot_cams = get_plot_cams_list(data, country=short_form)
    elif state:
        plot_cams = get_plot_cams_list(data, state=short_form)


    for i in range(data_points): 
        data_to_use = useful_data[start_samples[i]:end_samples[i]][plot_cams] 
        daily_counts[i] = np.max(np.sum(data_to_use[plot_cams], axis=1))


    """people counts -----------------------------------------------------------------
    """

    col = "pedestrian_count"
    people = "combined_csv_pedestrians_4-1_8-1.csv"
    colap = pd.read_csv('processed_people.csv')
    data = pd.read_csv(people)
    data = data.fillna(0)

    # remove the data of March 31
    useful_data = colap[1:]
    data_points = len(useful_data)

    start_samples = []
    end_samples = []
    plot_dates = []

    for i in range(data_points):
        start_samples.append(1*i)
        end_samples.append(min(1*(i+1), data_points))


    for i in range(data_points):
       plot_dates.append(np.array(useful_data[start_samples[i]:start_samples[i]+1]["date_keys"])[0])

    daily_counts_people = np.zeros((data_points))

    if country:
        plot_cams = get_plot_cams_list(data, country=short_form)
    elif state:
        plot_cams = get_plot_cams_list(data, state=short_form)

    for i in range(data_points): 
        data_to_use = useful_data[start_samples[i]:end_samples[i]][plot_cams] 
        daily_counts_people[i] = np.max(np.sum(data_to_use[plot_cams], axis=1))


    ax.set(title=place_to_use)

    colors, markers=color_list(plot_dates[1:-1], date1=date1, date2=date2, date3=date3, date4=date4)

    for i in range(len(daily_counts)):
        ax.scatter(daily_counts[i], daily_counts_people[:-2][i], color=colors[i], marker=markers[i])
    ax.set_xlabel('Vehicle count')
    ax.set_ylabel('People count')


    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='r', markersize=9),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='g', markersize=9),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='b', markersize=9),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='black', markersize=9),
                        Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='darkorchid', markersize=9),
                        ]

    dates = [date1, date2, date3, date4]
    legend = []
    prevdate = '04-01 ' 
    for each in dates:
        if each!=None:
            legend+=[str(prevdate + '-- ' + each)]
            prevdate=each

    legend+=[str(prevdate + '-- ' + '08-01 ')]

    ax.legend(legend_elements, legend)

    plt.title(place_to_use)
    plt.show()



























