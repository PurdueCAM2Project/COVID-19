import json

infile = '2_resized.json'
outfile = '2_resized_visual.json'

with open(infile, "r") as read_file:
    data = json.load(read_file)

formatted_detections = dict()
formatted_detections['object'] = []

for each in data.values():
    box = dict()
    box['bndbox'] = dict()
    box['bndbox']['xmin'] = each[0]
    box['bndbox']['ymin'] = each[1]
    box['bndbox']['xmax'] = each[2]
    box['bndbox']['ymax'] = each[3]

    formatted_detections['object'].append(box)

print(formatted_detections)
formatted_detections
with open(outfile, "w+") as out_file:
    out_file.write(json.dumps(formatted_detections))