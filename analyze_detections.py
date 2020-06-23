import json
import matplotlib.pyplot as plt

filename = 'person_detections'

with open(filename, 'r') as detections:
    d = json.load(detections)

print(d.keys())
for each in d.keys():
    print((d[each].keys()))
all_counts = dict()
for cam_id in d.keys():
    print('cam_id', cam_id)
    counts = dict()
    for image_name in d[cam_id]:
        counts[image_name] = 0
        for detection in d[cam_id][image_name]:
            if float(detection) > 0.3:
                counts[image_name] += 1
    all_counts[cam_id] = counts
#print(counts)

for key in all_counts.keys():
    print(key)
    l = []
    for each in all_counts[key]:
        l.append(all_counts[key][each])
    print(l)
    plt.plot(l)
    plt.savefig(key.split('/')[0])
