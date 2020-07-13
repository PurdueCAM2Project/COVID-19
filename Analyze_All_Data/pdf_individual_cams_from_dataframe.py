'''
Written by Subhankar and Minghao
Please do let us know in case of doubts
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy.stats import norm
import os
import sys
from numpy import sqrt, exp, pi

input_dir = './csv_files'
filename = 'July7_1035_beach_video_person'
filename = filename + '.csv'
STEPSIZE = 1000
NUMERIC_PRECISION = 1e-6
# check usage of NUMERIC_PRECISION where the standard deviation is being calculated
SAVEPATH = 'figures'
cam_type = "image" if "image" in filename.split('_') else "video"

location_key_dictionary = {'0DYhy8t7QU/': "Florida-USA", '11de6btHVN/': "Virgin Islands-USA", 'MpnI0K2MYW/': "Florida-USA",
                           'ov7H2byjWc/': "Virgin Islands-USA", 'nEVWDUiTdT/': "Eindhoven-The Netherlands"}

colors = ['r', 'ro', 'b', 'bo', 'k', 'ko']

plt.rcParams["figure.figsize"] = (8,6)


if not os.path.isdir(SAVEPATH):
    os.mkdir(SAVEPATH)

def get_int_from_datetime_object(date):
    return 10000*(date.year)+100*(date.month)+date.day


def get_label_from_date_in_integer_form(start_date, end_date):
    start_date_ = str(start_date)
    end_date_ = str(end_date)
    return start_date_[:4]+'-'+start_date_[4:6]+'-'+start_date_[6:-2] + ' to ' + end_date_[:4]+'-'+end_date_[4:6]+'-'+end_date_[6:-2]


# splitting the dates into three parts
# separation date 1 and 2 are the two boundaries at which we separate
separation_date_1 = '2020-04-29'  # please maintain this format for all dates
separation_date_2 = '2020-05-29'
separation_date_1 = get_int_from_datetime_object(
    datetime.strptime(separation_date_1, '%Y-%m-%d'))
separation_date_2 = get_int_from_datetime_object(
    datetime.strptime(separation_date_2, '%Y-%m-%d'))

with open(os.path.join(input_dir, filename), 'r') as f:
    dataframe = pd.read_csv(f)

cam_ids = list(pd.unique(dataframe['cam_id']))
detection_type = dataframe.columns[-1]
cameras = np.asarray(dataframe['cam_id'])
detections = dataframe[detection_type]

PLOT_LIMITED_CAMS = False
if PLOT_LIMITED_CAMS:
    print(f"{len(cam_ids)} cameras present.")
    n_cams_to_plot = int(
        input("How many cameras do you want to plot the data for.\n"))
    cam_ids = cam_ids[:n_cams_to_plot]

dates = dataframe['date']
print("Data present for", len(set(dates)), "dates.")

detections_per_camera = dict()
print("Preprocessing the data ....")
for camid in tqdm(cam_ids):
    relevant_dates = np.array(list(dates[cameras == camid]))
    relevant_detections = detections[cameras == camid]
    converted_dates = np.zeros(len(relevant_dates))
    detections_per_camera[camid] = dict()
    for i in range(len(relevant_dates)):
        date = relevant_dates[i]
        date_ = datetime.strptime(date, '%Y-%m-%d')
        date_ = get_int_from_datetime_object(date_)
        converted_dates[i] = date_

    unique_dates = np.asarray(list(set(converted_dates)))
    first_set_of_unique_dates = unique_dates[unique_dates < separation_date_1]
    second_set_of_unique_dates = unique_dates[(
        unique_dates >= separation_date_1) & (unique_dates < separation_date_2)]
    third_set_of_unique_dates = unique_dates[unique_dates >= separation_date_2]

    first_set_of_detections = np.zeros(len(first_set_of_unique_dates))
    second_set_of_detections = np.zeros(len(second_set_of_unique_dates))
    third_set_of_detections = np.zeros(len(third_set_of_unique_dates))

    for j in range(len(first_set_of_unique_dates)):
        unique_date = first_set_of_unique_dates[j]
        first_set_of_detections[j] = np.max(
            relevant_detections[converted_dates == unique_date])

    for j in range(len(second_set_of_unique_dates)):
        unique_date = second_set_of_unique_dates[j]
        second_set_of_detections[j] = np.max(
            relevant_detections[converted_dates == unique_date])

    for j in range(len(third_set_of_unique_dates)):
        unique_date = third_set_of_unique_dates[j]
        third_set_of_detections[j] = np.max(
            relevant_detections[converted_dates == unique_date])

    if len(first_set_of_unique_dates):
        start_date_1 = np.min(first_set_of_unique_dates)
        end_date_1 = np.max(first_set_of_unique_dates)
        label = get_label_from_date_in_integer_form(start_date_1, end_date_1)
        detections_per_camera[camid][label] = first_set_of_detections

    if len(second_set_of_unique_dates):
        start_date_2 = np.min(second_set_of_unique_dates)
        end_date_2 = np.max(second_set_of_unique_dates)
        label = get_label_from_date_in_integer_form(start_date_2, end_date_2)
        detections_per_camera[camid][label] = second_set_of_detections

    if len(third_set_of_unique_dates):
        start_date_3 = np.min(third_set_of_unique_dates)
        end_date_3 = np.max(third_set_of_unique_dates)
        label = get_label_from_date_in_integer_form(start_date_3, end_date_3)
        detections_per_camera[camid][label] = third_set_of_detections

print("Generating plots ....")
useful_camids = list(detections_per_camera.keys())
for camid in tqdm(useful_camids):
    cam_id = camid[:-1]
    detections_in_camera = detections_per_camera[camid]
    date_intervals = list(detections_in_camera.keys())
    xmin = np.inf
    xmax = -np.inf
    means = dict()
    standard_deviations = dict()
    final_detections = dict()
    for date_interval in date_intervals:
        detections_array = detections_per_camera[camid][date_interval]
        mu = np.mean(detections_array)
        # deal with the case when there is only one date sample leading to a zero variance and divide by zero warnings
        stddev = np.std(detections_array)+NUMERIC_PRECISION
        if mu-3*stddev < xmin:
            xmin = mu-3*stddev
        if mu+3*stddev > xmax:
            xmax = mu+3*stddev
        means[date_interval] = mu
        standard_deviations[date_interval] = stddev
        final_detections[date_interval] = detections_array

    xmin = max(0, xmin)
    # Take only from the positive side
    x = np.linspace(xmin, xmax, STEPSIZE)
    max_height = 0
    for k in range(len(date_intervals)):
        date_interval = date_intervals[k]
        mu = means[date_interval]
        sigma = standard_deviations[date_interval]
        p = (1/sqrt(2*pi*sigma**2)) * \
            (exp(-((x-mu)**2)/(2*sigma**2))+exp(-((x+mu)**2)/(2*sigma**2)))
        height = np.max(p)
        if height > max_height:
            max_height = height

    for k in range(len(date_intervals)):
        date_interval = date_intervals[k]
        mu = means[date_interval]
        sigma = standard_deviations[date_interval]
        p = (1/sqrt(2*pi*sigma**2)) * \
            (exp(-((x-mu)**2)/(2*sigma**2))+exp(-((x+mu)**2)/(2*sigma**2)))
        # height = np.max(p)
        # if height>max_height:
        #     max_height = height
        plt.plot(final_detections[date_interval], final_detections[date_interval]
                 * 0 + k*max_height/10, colors[2*k+1], label='detections for '+date_interval)
        plt.plot(x, p, colors[2*k], label="PDF for "+date_interval)

    # plt.legend(date_intervals)
    plt.legend()
    if camid in location_key_dictionary.keys():
        # plot_description = plt.title(f"Folded Gaussian PDF for camera {cam_id} in {location_key_dictionary[camid]}")
        plot_description = f"Folded Gaussian PDF for {cam_type} camera {cam_id} in {location_key_dictionary[camid]}"
    else:
        plot_description = f"Folded Gaussian PDF for {cam_type} camera {cam_id}"
        # plt.title(f"Folded Gaussian PDF for camera {cam_id}")
    plt.title(plot_description)
    name_to_save = plot_description.replace(" ", "_")+'.png'
    # full_save_name = os.path.join(SAVEPATH, name_to_save)
    full_save_name = os.path.join(SAVEPATH, name_to_save)
    plt.ylabel("Probability Density")
    plt.xlabel("number of people")
    plt.savefig(full_save_name)
    plt.close()
