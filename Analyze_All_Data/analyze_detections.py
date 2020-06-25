import json
import matplotlib.pyplot as plt

filename = 'results/vehicle_detections.json'

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
#print(counts)

for key in all_counts.keys():
    l = []
    for each in all_counts[key]:
        l.append(all_counts[key][each])
    plt.plot(l, label=key)
    plt.savefig('plots/'+key.split('/')[0])