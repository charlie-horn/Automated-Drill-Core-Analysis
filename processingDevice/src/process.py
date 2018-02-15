

##--------------Libraries----------------

from PIL import Image
import numpy as  np
import cv2
from matplotlib import pyplot as plt
import subprocess as sub
import sys
import os
import math
import re

##--------------Subroutines----------------

def svgToCoords(path):

	in_path = path
	out_path = "os.path.splitext(path)[0]" + "_coordinates.txt"

	in_f = open(in_path, 'r')
	out_f = open(out_path, 'w')

	type_check = re.compile(r"\<(<type>\w+)"
	path_pattern = re.compile(r"\<(<type>\w+) d="M 545.904775,351.973887 A620.572292,620.572292 0.000000 0,1 547.188710,380.040344" fill="none" stroke ="red" stroke-width="1" />"
	line_pattern = 

	first = True

	for i, line in f.readlines():

		continue if i < 5

		# For now skip paths, only check lines
		continue if line.startswith

		path_match = re.match(path_pattern, line)
		line_match = re.match(line_pattern, line)

		continue if path_match.groups('type') != 'path' && line_match.groups('type') != 'line'

		if line_type



def getLine(edge):
	x1 = edge[0][0]
	y1 = edge[0][1]
	x2 = edge[1][0]
	y2 = edge[1][1]
	
	line = []

	#if y1 > y2:
	#	x_temp = x1
	#	x1 = x2
	#	x2 = x_temp
	#	y_temp = y1
	#	y1 = y2
	#	y2 = y_temp

	m, b = getMB(x1,y1,x2,y2)

	for y in range(int(y1),int(y2)+1):
		x = int((y-b)/m)
		line.append((x,y))
	return line

def getMB(x1,y1,x2,y2):
	rise = y2 - y1
	run = x2 - x1
	slope = rise/run
	intercept = y1 - slope*x1
	return slope, intercept

def click(event, x, y, flags, params):
	global points

	if event == cv2.EVENT_LBUTTONUP:
		points.append((x, y))
		RGB = pixels[x,y]
		print (x,y)
		print RGB
		rgbs.append(RGB)


##--------------Classes----------------

# class Feature {}

# class Line(Feature) {}

# class Ellipse(Feature) {}


##--------------Program----------------

# Copy to a new image so the original doesn't get alterred

image_path = sys.argv[1]
new_image_path = os.path.splitext(image_path)[0]+"_alterred"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, new_image_path])

# Read in new image for editing

image = cv2.imread(new_image_path)
image_height = image.shape[0]
min_line_length = image_height/3

# Apply original ELSDc with no modification to parameters

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_image_path = os.path.splitext(new_image_path)[0]+".pgm"
cv2.imwrite(grey_image_path, grey_image)
home_dir = os.path.expanduser("~")
elsdc_path = home_dir + "/mthe493/catkin_ws/src/elsdc/src/elsdc"
sub.call([elsdc_path, grey_image_path])

# Find longest vertical polygons from output

currentDir = os.getcwd()
polygon_file = open(currentDir + "/out_polygon.txt", 'r')
polygons = {}
long_edges = []
for polygon in polygon_file.readlines():
	split_line = polygon.split(' ')
	index = split_line[0]
	length = split_line[1] #Amount or vertices in the polygon
	polygons[index] = []
	i =  2 #Start from the first x coordinate
	while i < len(split_line)-1:
		(x,y) = float(split_line[i]), float(split_line[i+1])
		polygons[index].append((x,y))
		i += 2

		current_polygon = polygons[index]
		flagged_as_long = False
		for ref_vertex in current_polygon:
			for comp_vertex in current_polygon:
				x_dist = ref_vertex[0] - comp_vertex[0]
				y_dist = ref_vertex[1] - comp_vertex[1]
				side_length = math.sqrt(x_dist**2+y_dist**2)
				if side_length >= min_line_length:
					long_edges.append((ref_vertex, comp_vertex))
					flagged_as_long = True
					break
			if flagged_as_long:
				break

long_lines = []

# Determine if the lines are far left or far right

left_line = []
for line in long_lines:
	furthest_left = True
	for point in line:
		for compare_line in long_lines:
			for compare_point in compare_line:
				if point[1] == compare_point[1]:
					if compare_point[0] < point[0]:
						furthest_left = False
						break
					else:
						break
			if not furthest_left:
				break
		if not furthest_left:
			break
	if furthest_left:
	 left_line = line

for edge in long_edges:
	line = getLine(edge)
	long_lines.append(line)

# Draw long lines on image

for line in long_lines:
	for point in line:
		for i in range(0,point[0]):
			try:
				image[point[1],i] = [0,255,0]
			except(IndexError):
				continue

cv2.imwrite(new_image_path, image)
cv2.imshow('Added Green', image)
cv2.waitKey(0)
