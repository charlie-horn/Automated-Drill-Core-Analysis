#!/bin/bash

image=$(zenity --file-selection --title="circle2img3.pgm")
outputDir=$(zenity --file-selection --directory --title="Choose output folder")
defaultOutputFolder=$(date '+%d_%m_%Y_%H_%M_%S')
outputFolder=$(zenity --entry --title "Outlier Fraction" --text="Pick a name for the output folder." --entry-text=$defaultOutputFolder)

cd $outputDir
mkdir $outputFolder
cd $outputFolder
cp $image .
image_name=$(basename "$image")
separator="/"
echo $image_name
new_image_path=$PWD$separator$image_name

python $HOME/mthe493/catkin_ws/src/processingDevice/src/process.py $new_image_path
