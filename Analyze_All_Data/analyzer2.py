import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
from os import path, mkdir
import numpy as np
import sys
sys.path.append("../")
sys.path.append("./")


class Analyzer:
    def __init__(self):
        # self.filename = 'all_data.csv'
        # if path.exists(self.filename):
        #     self.df = pd.read_csv(self.filename)
        # else:

        self.df_person = pd.DataFrame(columns=(
            'date', 'cam_id', 'night', 'dense', 'type', 'place', 'pedestrian_count'))

        self.df_vehicles = pd.DataFrame(columns=(
            'date', 'cam_id', 'night', 'dense', 'type', 'place', 'vehicle_count'))

    def merge(self, dict1, dict2):
        return (dict2.update(dict1))

    def load_json(self, json_file):
        with open(json_file, 'r') as infile:
            text = infile.read()
            d = json.loads(text)
        return d

    def consolidate_individual_video_detections(self, filenames):
        """
        if all video detections separate, first merge into one dictionary
        :param filenames: list of filenames
        :return: merged dict {cam_id: {date: {frame:count, frame: count}}, cam_id:...}
        """
        with open(filenames[0], 'r') as file:
            merged_dict = json.load(file)

        for each in filenames[1:]:
            with open(each, 'r') as file:
                d = json.load(file)
                print(d.keys())
                print(len(d.keys()))
                self.merge(merged_dict, d)

        print(merged_dict.keys())
        print(len(merged_dict.keys()))
        return merged_dict

    def simplify_video_detections(self, video_dict: dict, filename_to_save=None, day_night_dictionary=None, conf_threshold=0.3):
        """
        function to parse video detections to max_video detections (same format as image detections)
        input: {cam_id: {date: {frame:count, frame: count}}}
        detection_type = ['vehicles','people']
        :return: simplified dict {cam_id: {date: count, date: count}}
        """
        simplified_dict = dict()
        if day_night_dictionary != None:
            day_night_dict = dict()
        for cam_id in video_dict:
            simplified_dict[cam_id] = dict()
            if day_night_dictionary != None:
                day_night_dict[cam_id] = dict()
            for date_time in video_dict[cam_id]:
                max_count = 0
                for frame in video_dict[cam_id][date_time]:
                    count = 0
                    for detection in video_dict[cam_id][date_time][frame]:
                        if float(detection) > conf_threshold:
                            count += 1
                    if count > max_count:
                        max_count = count
                        if day_night_dictionary != None:
                            day_night_pred = day_night_dictionary[cam_id][date_time][frame]
                simplified_dict[cam_id][date_time] = max_count
                if day_night_dictionary != None:
                    day_night_dict[cam_id][date_time] = day_night_pred

        if filename_to_save != None:
            with open(filename_to_save, 'w+') as simple_fp:
                simple_fp.write(json.dumps(simplified_dict))

        if day_night_dictionary != None:
            return simplified_dict,  day_night_dict
        else:
            return simplified_dict

    def simplify_image_results(self, image_dict: dict, filename_to_save=None, confidence_threshold=0.3, object_='person'):
        '''
        simplify image results to the number of counts in each image
        '''
        simplified_dict = dict()
        for cam_id in image_dict.keys():
            simplified_dict[cam_id] = dict()
            for img_url in image_dict[cam_id]:
                detections = image_dict[cam_id][img_url]
                if object_ != 'person':
                    count = len(detections.keys())
                else:
                    count = 0
                    confidences = list(detections.keys())
                    for confidence in confidences:
                        if float(confidence) > confidence_threshold:
                            count += 1
                simplified_dict[cam_id][img_url] = count
        return simplified_dict

    def normalize_simplified_dict(self, in_dict):
        d = in_dict.copy()
        print(d)

        for cam_id in d.keys():
            try:
                largest_value = max(d[cam_id].values())
                for date in d[cam_id]:
                    d[cam_id][date] = d[cam_id][date]/largest_value
            except ValueError:
                # largest_value = float('inf')
                for date in d[cam_id]:
                    d[cam_id][date] = 0
        return d

    def add_results_df(self, results_dict, obj='person', day_night_dict=None, cam_type='video', filename='detections', savedir='dataframes/'):
        """
        function to parse simplified json results into dataframe
        if video data, results must be simplified first using simplify_video_detections
        results_dict = either a video or image dictionary of results
        cam_type = ['video', 'image']
        obj = ['vehicle', 'person']
        file is saved as "savedir/filename_cam_type_obj.csv"
        :return: None
        """
        p = re.compile('\d\d\d\d-\d\d-\d\d')
        if not path.isdir(savedir):
            mkdir(savedir)
        save_path = path.join(savedir, filename+'_'+cam_type+'_'+obj+'.csv')
        if obj == 'vehicle':
            key = 'vehicle_count'
            frames = [self.df_vehicles]
        elif obj == 'person':
            key = 'pedestrian_count'
            frames = [self.df_person]

        size = len(results_dict.keys()) - 1

        # save all dictionaries
        record = []
        if cam_type != 'video':
            for i, cam_id in enumerate(results_dict):
                if len(results_dict[cam_id]) > 0:
                    for img_url in results_dict[cam_id]:
                        data = {'date': pd.to_datetime(p.search(img_url).group(
                            0)), 'cam_id': cam_id, 'type': cam_type, key: None}
                        data[key] = results_dict[cam_id][img_url]

                        if day_night_dict != None:
                            data['night'] = day_night_dict[cam_id][img_url]
                        record.append(data)
                    print(f"  {i}/{size}\r", flush=True, end="")

            # build the data frame
            frames.append(pd.DataFrame.from_records(record))
            self.df_person = pd.concat(frames, sort=False)
            # self.df_person.to_csv(obj+'_image.csv')
            self.df_person.to_csv(save_path)

            # display head and tail
            print(self.df_person.head(5))
            print(self.df_person.tail(5))
            return
            pass
        else:
            # builds dictionary record for videos
            for i, cam_id in enumerate(results_dict):
                if len(results_dict[cam_id]) > 0:
                    for date_time in results_dict[cam_id]:
                        data = {'date': pd.to_datetime(p.search(date_time).group(
                            0)), 'cam_id': cam_id, 'type': cam_type, key: None}
                        data[key] = results_dict[cam_id][date_time]

                        if day_night_dict != None:
                            data['night'] = day_night_dict[cam_id][date_time]
                        record.append(data)
                    print(f"  {i}/{size}\r", flush=True, end="")

            # build the data frame
            frames.append(pd.DataFrame.from_records(record))
            self.df_vehicles = pd.concat(frames, sort=False)
            # self.df_vehicles.to_csv(obj+'_video.csv')
            self.df_vehicles.to_csv(save_path)

            # display head and tail
            print(self.df_vehicles.head(5))
            print(self.df_vehicles.tail(5))
            return

    def plot_time_series(self):
        """
        @Todo: plot time series of each unique cam
        :return: graph
        """
        pass

    def easy_plot(self, video_simple_results):

        for key in video_simple_results.keys():
            l = []
            for each in video_simple_results[key]:
                l.append(video_simple_results[key][each])
            plt.plot(l, label=key)
            plt.legend()

        plt.show()

    def plot_car_detections(self, filename):

        with open(filename, 'r') as detections:
            d = json.load(detections)

        print(len(d.keys()))
        all_counts = dict()

        for cam_id in d.keys():
            print('cam_id', cam_id)
            counts = dict()
            for image_name in d[cam_id]:
                counts[image_name] = 0
                for detection in d[cam_id][image_name]:
                    # if float(detection) > 0.3:
                    counts[image_name] += 1
            all_counts[cam_id] = counts

        self.easy_plot(all_counts)


if __name__ == "__main__":

    """
    example usage
    """
    a = Analyzer()

    image_dictionary = a.load_json('results/July6_1030_vehicle_image.json')
    dn_dictionary = a.load_json('results/July6_1030_day_night.json')
    obj = 'vehicle'
    cam_type = 'image'
    filename = 'July6_1030'
    savedir = 'dataframes'
    conf_threshold = 0.3

    image_keys = set(image_dictionary.keys())
    dn_keys = set(dn_dictionary.keys())
    common_keys = dn_keys.intersection(image_keys)
    bottleneck_n = len(common_keys)
    if bottleneck_n == 0:
        print("No common keys in the image and day night JSON")
        print("Please check your data")
        print("Exiting....")
        sys.exit(0)

    mini_image_dictionary = dict()
    mini_dn_dictionary = dict()

    print(f"{len(image_keys)} keys present in the image JSON")
    print(f"{len(dn_keys)} keys present in the day night JSON")
    print(f"{bottleneck_n} common keys present.")
    for key in common_keys:
        mini_image_dictionary[key] = image_dictionary[key]
        mini_dn_dictionary[key] = dn_dictionary[key]

    if cam_type == 'image':
        mini_image_results_people = a.simplify_image_results(
            mini_image_dictionary, object_=obj, confidence_threshold=conf_threshold)
        a.add_results_df(mini_image_results_people, day_night_dict=mini_dn_dictionary,
                         cam_type='image', obj=obj, filename=filename, savedir=savedir)
    elif cam_type == 'video':
        mini_image_results_people, mini_dn_dictionary = a.simplify_video_detections(
            mini_image_dictionary, day_night_dictionary=mini_dn_dictionary, conf_threshold=conf_threshold)
        a.add_results_df(mini_image_results_people, day_night_dict=mini_dn_dictionary,
                         cam_type='video', obj=obj, filename=filename, savedir=savedir)
    