#!/bin/bash
file="datasets.zip"

if [ -f "$file" ];
then
   echo "File $FILE already downloaded."
else
   echo "File $FILE does not exist" && \
   wget -O datasets.zip http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
fi

unzip datasets.zip -d datasets