import json
import os
import glob

for each in glob.glob("jax_results/*.json"):
	print(each)
	infile = each
	outfile = each.split('.json')[0] + '_converted.json'


	with open(infile, "r") as read_file:
	    data = json.load(read_file)

	formatted_detections = dict()
	formatted_detections['object'] = []

	print(len(data.keys()))
	for key in data.keys():
		#print(key)
		if float(key) >= 0.3:
			each = data[key]
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