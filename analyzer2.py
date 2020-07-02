import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
from os import path, mkdir
import numpy as np

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
                if object_!='person':
                    count = len(detections.keys())
                else:
                    count = 0
                    confidences = list(detections.keys())
                    for confidence in confidences:
                        if float(confidence)>confidence_threshold:
                            count+=1
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

    def add_results_df(self, results_dict , obj='person', day_night_dict=None, cam_type='video', filename='detections', savedir='dataframes/'):
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
        if cam_type!='video':
            for i, cam_id in enumerate(results_dict):
                if len(results_dict[cam_id]) > 0:
                    for img_url in results_dict[cam_id]:
                        data = {'date': pd.to_datetime(p.search(img_url).group(
                            0)), 'cam_id': cam_id, 'type': cam_type, key: None}
                        data[key] = results_dict[cam_id][img_url]

                        if day_night_dict!=None:
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

                        if day_night_dict!=None:
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


    """
    plot person detections from raw video results
    """
    # # save raw results into one dict
    # video_results_files = ['../person_detections_video']
    # merged_dict = a.consolidate_individual_video_detections(video_results_files)

    # # simplify raw dict
    # simple_video_results = a.simplify_video_detections(merged_dict, 'simple_video_detections_person')

    # # normalize
    # simple_video_results_normalized = a.normalize_simplified_dict(simple_video_results)

    # # plot
    # a.easy_plot(simple_video_results_normalized)


    """
    add json detections into dataframe
    """

    # simple_video_results_person = a.load_json('simple_video_detections_person')
    # a.add_results_df(simple_video_results_person, 'video', 'person')
    #
    # image_results_car = a.load_json('../vehicle_detections.json')
    # image_results_car_simple = a.simplify_video_detections(image_results_car, 'simple_image_detections_car')
    # a.add_results_df(image_results_car_simple, 'image', 'vehicle')

    image_results_people = a.load_json('results/person_detections_0_mini.json')
    dn_detections_images = a.load_json('results/day_night_images_mini.json')
    image_results_people = a.simplify_image_results(image_results_people, object_='person')
    a.add_results_df(image_results_people, day_night_dict=dn_detections_images, cam_type='image', obj='person', filename='trial', savedir='dataframes')

    image_results_vehicles = a.load_json('results/vehicle_detections_mini.json')
    image_results_vehicles = a.simplify_image_results(image_results_vehicles, object_='vehicle')
    a.add_results_df(image_results_vehicles, day_night_dict=dn_detections_images, cam_type='image', obj='vehicle', filename='trial1', savedir='dataframes')

    video_results_people = a.load_json('results/person_detections_video_mini.json')
    dn_detections_videos = a.load_json('results/day_night_video_detections_mini.json')
    imr, dnr = a.simplify_video_detections(video_results_people, day_night_dictionary=dn_detections_videos)
    a.add_results_df(imr, day_night_dict=dnr, obj='person', cam_type='video', filename='trial2', savedir='dataframes')    