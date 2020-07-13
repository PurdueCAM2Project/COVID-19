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


input_dir = './'
filename = 'July7_1035_beach_video_person'
filename = filename +'.csv'
STEPSIZE = 1000
NUMERIC_PRECISION = 1e-6
# check usage of NUMERIC_PRECISION where the standard deviation is being calculated
SAVEPATH = 'figures'
if not os.path.isdir(SAVEPATH):
    os.mkdir(SAVEPATH)
SHOW_PLOT = True


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
all_converted_dates = np.zeros(len(dates))
print("Data present for", len(set(dates)), "dates.")
print("Converting the dates ....")
for i in tqdm(range(len(all_converted_dates))):
    date = dates[i]
    date = get_int_from_datetime_object(datetime.strptime(date, '%Y-%m-%d'))
    all_converted_dates[i] = date

# all_converted_dates = all_converted_dates.astype(np.int)
unique_dates = np.array(list(set(all_converted_dates)))
detections_datewise = dict()


dates_1 = unique_dates[unique_dates < separation_date_1]
if len(dates_1):
    start_date_1 = np.min(dates_1)
    stop_date_1 = np.max(dates_1)
    label_1 = get_label_from_date_in_integer_form(start_date_1, stop_date_1)
    counts_1 = np.zeros_like(dates_1)
    for i in range(len(dates_1)):
        date = dates_1[i]
        relevant_camera_ids = np.array(
            dataframe['cam_id'][all_converted_dates == date])
        relevant_detections = detections[all_converted_dates == date]
        unique_camera_ids = np.array(list(set(relevant_camera_ids)))
        for camid in unique_camera_ids:
            n_detections = np.max(
                relevant_detections[relevant_camera_ids == camid])
            counts_1[i] += n_detections
    detections_datewise[label_1] = counts_1

dates_2 = unique_dates[(unique_dates >= separation_date_1)
                       & (unique_dates < separation_date_2)]
if len(dates_2):
    start_date_2 = np.min(dates_2)
    stop_date_2 = np.max(dates_2)
    label_2 = get_label_from_date_in_integer_form(start_date_2, stop_date_2)
    counts_2 = np.zeros_like(dates_2)
    for i in range(len(dates_2)):
        date = dates_2[i]
        relevant_camera_ids = np.array(
            dataframe['cam_id'][all_converted_dates == date])
        relevant_detections = detections[all_converted_dates == date]
        unique_camera_ids = np.array(list(set(relevant_camera_ids)))
        for camid in unique_camera_ids:
            n_detections = np.max(
                relevant_detections[relevant_camera_ids == camid])
            counts_2[i] += n_detections
    detections_datewise[label_2] = counts_2


dates_3 = unique_dates[unique_dates >= separation_date_2]
if len(dates_3):
    start_date_3 = np.min(dates_3)
    stop_date_3 = np.max(dates_3)
    label_3 = get_label_from_date_in_integer_form(start_date_3, stop_date_3)
    counts_3 = np.zeros_like(dates_3)
    for i in range(len(dates_3)):
        date = dates_3[i]
        relevant_camera_ids = np.array(
            dataframe['cam_id'][all_converted_dates == date])
        relevant_detections = detections[all_converted_dates == date]
        unique_camera_ids = np.array(list(set(relevant_camera_ids)))
        for camid in unique_camera_ids:
            n_detections = np.max(
                relevant_detections[relevant_camera_ids == camid])
            counts_3[i] += n_detections
    detections_datewise[label_3] = counts_3

date_intervals = list(detections_datewise.keys())
xmin = np.inf
xmax = -np.inf
means = dict()
standard_deviations = dict()
for date_interval in date_intervals:
    detections_array = detections_datewise[date_interval]
    mu = np.mean(detections_array)
    # deal with the case when there is only one date sample leading to a zero variance and divide by zero warnings
    stddev = np.std(detections_array)+NUMERIC_PRECISION
    if mu-3*stddev < xmin:
        xmin = mu-3*stddev
    if mu+3*stddev > xmax:
        xmax = mu+3*stddev
    means[date_interval] = mu
    standard_deviations[date_interval] = stddev

xmin = max(0, xmin)
# Take only from the positive side
x = np.linspace(xmin, xmax, STEPSIZE)
for date_interval in date_intervals:

    # p = norm.pdf(x, means[date_interval],
    #              standard_deviations[date_interval])
    mu = means[date_interval]
    sigma = standard_deviations[date_interval]
    p = (1/sqrt(2*pi*sigma**2)) * \
        (exp(-((x-mu)**2)/(2*sigma**2))+exp(-((x+mu)**2)/(2*sigma**2)))
    plt.plot(x, p)

plt.legend(date_intervals)
plt.title(f"Folded Gaussian PDF for {len(cam_ids)} cameras in {filename.split('.')[0].replace('_', ' ')}")
name_to_save = filename.split('.')[0]+'.png'
full_save_name = os.path.join(SAVEPATH, name_to_save)
plt.ylabel("Probability Density")
plt.xlabel("Number of People")
plt.savefig(full_save_name)
if SHOW_PLOT:
    plt.show()
plt.close()
