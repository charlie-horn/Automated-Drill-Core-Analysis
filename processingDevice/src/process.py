

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
import shutil
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier

##--------------Subroutines----------------

def svgToCoords(svg_path):
	trunc = os.path.splitext(svg_path)[0]
	txt_path = trunc + ".txt"

	shutil.copyfile(svg_path, txt_path)

	in_path = txt_path
	out_path = trunc + "_coordinates.txt"

	in_f = open(in_path, 'r')
	out_f = open(out_path, 'w')
	
	f = in_f.readlines()

	line_pattern = re.compile(r'<(?P<type>\w{4}) x1=\"(?P<x1>-?\d+\.?\d*)\" y1=\"(?P<y1>-?\d+\.?\d*)\" x2=\"(?P<x2>-?\d+\.\d*?)\" y2=\"(?P<y2>-?\d+\.?\d*)\".*')
	path_pattern = re.compile(r'<(?P<type>\w{4}) d=\"M (?P<x1>-?\d+\.?\d*),(?P<y1>-?\d+\.?\d*) A(?P<r1>\d+\.?\d*),(?P<r2>\d+\.?\d*) (?P<rotation>-?\d+\.?\d*) (?P<arc>\d{1}),(?P<sweep>\d{1}) (?P<x2>-?\d+\.?\d*),(?P<y2>-?\d+\.?\d*)\".*')
	
	for i, line in enumerate(f):

		if line.startswith("<line"):
			
			line_match = re.match(line_pattern, line)

			x1 = line_match.group('x1')
			y1 = line_match.group('y1')
			x2 = line_match.group('x2')
			y2 = line_match.group('y2')

			start = complex(float(x1),float(y1))
			end = complex(float(x2),float(y2))

			coords = getLineCoords(start, end)
			if coords != []:
				out_f.write(str(coords))
				out_f.write("\n")
			
		elif line.startswith("<path"):

			path_match = re.match(path_pattern, line)
			x1 = path_match.group('x1')
			y1 = path_match.group('y1')
			r1 = path_match.group('r1')
			r2 = path_match.group('r2')
			rotation = float(path_match.group('rotation'))
			arc = int(path_match.group('arc'))
			sweep = int(path_match.group('sweep'))
			x2 = path_match.group('x2')
			y2 = path_match.group('y2')

			start = complex(float(x1),float(y1))
			radius = complex(float(r1),float(r2))
			end = complex(float(x2),float(y2))

			coords = getPathCoords(start, radius, rotation, arc, sweep, end)
			out_f.write(str(coords))
			out_f.write("\n")

	in_f.close()
	out_f.close()

def getLineCoords(start, end):
	path = Line(start, end)
	coords = []

	slope = getSlope(start, end)
	if abs(slope) > MAX_SLOPE:
		return coords

	for i in drange(0, 1, 0.1):
		coords.append(path.point(i))
	return coords

def getPathCoords(start, radius, rotation, arc, sweep, end):
	path = Arc(start, radius, rotation, arc, sweep, end)
	coords = []

	for i in drange(0, 1, 0.01):
		coords.append(path.point(i))
	
	return coords

def drange(start, stop, step):
    while start < stop:
		yield start
		start += step

def drawLine(coords, image, colour=[0,255,0]):
	for point in coords.split(", "):
		if point.startswith("["):
			point = point[2:-1]
		elif point.endswith("]"):
			point = point[1:-2]
		else:
			point = point[1:-1]

		point = complex(point)
		x = point.real
		y = point.imag
		try:
			image[y,x] = colour
		except(IndexError):
			continue
	return

def getSlope(start, end):
	rise = end.imag - start.imag
	run = end.real - start.real
	slope = rise/run
	return slope

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


##----------Global Variables-----------

MAX_LINE_LENGTH = 2
MAX_SLOPE = 5

##--------------Program----------------

# Copy to a new image so the original doesn't get alterred

image_path = sys.argv[1]
new_image_path = os.path.splitext(image_path)[0]+"_alterred"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, new_image_path])

# Read in new image for editing

image = cv2.imread(new_image_path)

# Apply original ELSDc with no modification to parameters

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_image_path = os.path.splitext(new_image_path)[0]+".pgm"
cv2.imwrite(grey_image_path, grey_image)
home_dir = os.path.expanduser("~")
elsdc_path = home_dir + "/mthe493/catkin_ws/src/elsdc/src/elsdc"
sub.call([elsdc_path, grey_image_path])

# Convert output.svg to a txt file containing X,Y coordinates for all lines

svgToCoords(os.path.dirname(new_image_path)+"/output.svg")

# Draw all detected lines in a given colour

coords_file = os.path.dirname(new_image_path)+"/output_coordinates.txt"

cf = open(coords_file, 'r')

for i, line in enumerate(cf.readlines()):
	drawLine(line.rstrip("\n"), image)

cv2.imwrite(new_image_path, image)
cv2.waitKey(0)
