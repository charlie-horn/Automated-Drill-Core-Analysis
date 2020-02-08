#!/bin/bash

image=$(zenity --file-selection --title="Choose image")
dirname=$(dirname $0)
basename=$(basename $0)
separator="/"
pythonFileName="process.py"
fullPythonPath=$PWD$separator$dirname$separator$pythonFileName
outputDir=$PWD$separator$dirname$separator$"../output/"
defaultOutputFolder=$(date '+%d_%m_%Y_%H_%M_%S')
outputFolder=$(zenity --entry --title "Outlier Fraction" --text="Pick a name for the output folder." --entry-text=$defaultOutputFolder)

cd $outputDir
mkdir $outputFolder
cd $outputFolder
cp $image .
image_name=$(basename "$image")
separator="/"
new_image_path=$PWD$separator$image_name

python $fullPythonPath $new_image_path
