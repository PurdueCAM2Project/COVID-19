import json
import pandas as pd
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self):
        self.df = pd.DataFrame()

    def merge(self, dict1, dict2):
        return (dict2.update(dict1))

    def consolidate_individual_video_detections(self, filenames):
        """
        if all video detections separate, first merge into one dictionary
        :param filenames: list of filenames
        :return: merged dict {cam_id: {date: {frame:count, frame: count}}, cam_id:...}
        """
        with open (filenames[0], 'r') as file:
            merged_dict = json.load(file)

        for each in filenames[1:]:
            with open (each, 'r') as file:
                d = json.load(file)
                print(d.keys())
                print(len(d.keys()))
                self.merge(merged_dict, d)

        print(merged_dict.keys())
        return merged_dict

    def simplify_video_detections(self, video_dict: dict):
        """
        function to parse video detections to max_video detections (same format as image detections)

        input: {cam_id: {date: {frame:count, frame: count}}}
        :return: simplified dict {cam_id: {date: count, date: count}}
        """
        simplified_dict = dict()
        for cam_id in video_dict:
            simplified_dict[cam_id] = dict()

            for date_time in video_dict[cam_id]:
                max_count = 0
                for frame in video_dict[cam_id][date_time]:
                    count = 0
                    for detection in video_dict[cam_id][date_time][frame]:
                        if float(detection) > 0.3:
                            count += 1
                    if count > max_count:
                        max_count = count
                simplified_dict[cam_id][date_time] = max_count

        return simplified_dict

    def normalize_simplified_dict(self, in_dict):
        d = in_dict.copy()
        for cam_id in d:
            s = sum(d[cam_id].values())
            for date in d[cam_id]:
                d[cam_id][date] = d[cam_id][date]/s

        return d


    def add_results_df(self, results_dict, cam_type, object):
        """
        function to parse simplified json results into dataframe
        if video data, results must be simplified first using simplify_video_detections

        results_dict = either a video or image dictionary of results
        cam_type = ['video', 'image']
        object = ['car', 'person']
        :return: None
        """

        pass

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
        plt.show()
            #plt.savefig('plots/' + key.split('/')[0])


    def plot_car_detections(self, filename):

        with open(filename, 'r') as detections:
            d = json.load(detections)

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
    video_results_files = ['../person_detections_video']

    a = Analyzer()

    # with open(video_results_file, 'r') as infile:
    #     video_results = json.load(infile)
    merged_dict = a.consolidate_individual_video_detections(video_results_files)

    simple_video_results = a.simplify_video_detections(merged_dict)
    simple_video_results_normalized = a.normalize_simplified_dict(simple_video_results)
    # print(simple_video_results
    # with open('video_detections.json', 'w+') as outfile:
    #     outfile.write(json.dumps(simple_video_results))

    a.easy_plot(simple_video_results_normalized)
    # a.add_results_df(simple_video_results)
    # print(a.df)

