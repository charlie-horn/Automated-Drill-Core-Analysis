#!/bin/bash

image=$(zenity --file-selection --title="Choose image")
outputDir=$(zenity --file-selection --directory --title="Choose output folder")
defaultOutputFolder=$(date '+%d_%m_%Y_%H_%M_%S')
outputFolder=$(zenity --entry --title "Outlier Fraction" --text="Pick a name for the output folder." --entry-text=$defaultOutputFolder)

cd $outputDir
mkdir $outputFolder
cd $outputFolder
cp $image .

python $HOME/mthe494/catkin_ws/src/processingDevice/src/process.py $image
