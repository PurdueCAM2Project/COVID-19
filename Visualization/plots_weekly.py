import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from plotting_tools import *


 # dates need to be added here
useful_dates = [("April", 6), ("April", 13), ("April", 20), ("April", 27), ("May", 4),
                ("May", 11), ("May", 18), ("May", 25), ("June", 1), ("June", 8), ("June", 15),
                ("June", 22), ("June", 29), ("July", 6), ("July", 13), ("July", 20), ("July", 27), ("August", 1)]


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

disclude_people = ["CA", "DE", "NZ", "HK", "ES", "HR"]
disclude_vehicles = ["IT", "CH"]

for ax, place_to_use, short_form in zip(axes.flatten(), places_to_use, short_forms):
   

    col = "vehicle_count"
    cars = "combined_csv_vehicles_without_night_place_city_pedestriancount.csv"
    colap = pd.read_csv('processed_vehicles.csv')
    data = pd.read_csv(cars)
    data = data.fillna(0)

    # # dates need to be added here as well
    Dates = ["April 6", "April 13", "April 20", "April 27", "May 4",
             "May 11", "May 18", "May 25", "June 1", "June 8", "June 15", "June 22", "June 29", "July 6", "July 13", "July 20", "July 27", "August 1"]

    Dates_2 = ["April 1", "April 8", "April 15", "April 22", "April 29",
             "May 6", "May 13", "May 20", "May 27", "June 3", "June 10", "June 17", "June 24", "July 1", "July 8", "July 15", "July 22", "July 29"]

    # remove March 31 data
    useful_data = colap[1:]
    n_weeks = len(useful_dates)
    data_points = len(useful_data)

    start_samples = []
    end_samples = []
    plot_dates = []

    for i in range(n_weeks):
        start_samples.append(7*i)
        end_samples.append(min(7*(i+1), data_points))

    for i in range(n_weeks):
        plot_dates.append(np.array(useful_data[start_samples[i]:start_samples[i]+1]["date_keys"])[0])

    weekly_counts = np.zeros(len(plot_dates))

    plot_cams = get_plot_cams_list(data, country=short_form)

    for i in range(n_weeks): 
        data_to_use = useful_data[start_samples[i]:end_samples[i]][plot_cams] 
        weekly_counts[i] = np.max(np.sum(data_to_use[plot_cams], axis=1))



    #people counts -----------------------------------------------------------------

    col = "pedestrian_count"
    people = "combined_csv_pedestrians_without_night_place.csv"
    colap = pd.read_csv('processed_people.csv')
    data = pd.read_csv(people)
    data = data.fillna(0)


    # # dates need to be added here as well
    Dates = ["April 6", "April 13", "April 20", "April 27", "May 4",
             "May 11", "May 18", "May 25", "June 1", "June 8", "June 15", "June 22", "June 29", "July 6", "July 13", "July 20", "July 27"]

    useful_data = colap[1:]
    n_weeks = len(useful_dates)
    data_points = len(useful_data)

    start_samples = []
    end_samples = []
    plot_dates = []


    for i in range(n_weeks):
        start_samples.append(7*i)
        end_samples.append(min(7*(i+1), data_points))

    for i in range(n_weeks):
        plot_dates.append(np.array(useful_data[start_samples[i]+1:start_samples[i]+2]["date_keys"])[0])

    weekly_counts_people = np.zeros(len(plot_dates))

    plot_cams = get_plot_cams_list(data, country=short_form)


    for i in range(n_weeks): 
        data_to_use = useful_data[start_samples[i]:end_samples[i]][plot_cams] 
        weekly_counts_people[i] = np.max(np.sum(data_to_use[plot_cams], axis=1))


    ax.set(title=place_to_use)

    if short_form not in disclude_vehicles:
        ax.bar(plot_dates, weekly_counts, align='edge', width=0.4, color='white', edgecolor='black')
        ax.set_ylabel('Number of Vehicles')

   
    if short_form not in disclude_people:
        ax3 = ax.twinx()

        ax3.bar(plot_dates, weekly_counts_people, align='edge', color='black', width=-0.4)
        ax3.set_ylabel('Number of People')
    

    ax.set_xticklabels(plot_dates, rotation=60)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
  
plt.title(place_to_use)
plt.show()



