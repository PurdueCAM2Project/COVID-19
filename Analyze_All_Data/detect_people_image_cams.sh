#!/bin/bash

if [ -x "$(command -v conda)" ]; then
  if [ "$(conda env list | grep ^ighodgao)" != "" ]; then 
      echo "Found Isha's environment at ANL."
      source activate ighodgao
   fi
fi

# This makes sure we have the required args
if [ "$#" -lt 3 ]; then
  echo "usage: ./downloader.sh start end"
  exit 1
fi

start_ind=$1
end_ind=$2
filen=$3

cd "COVID-19"
python3 detect_people_image_cams.py --start_index="$start_ind" --end_index="$end_ind" --filename="$filen"
