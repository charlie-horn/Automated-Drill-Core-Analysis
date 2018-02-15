

##--------------Libraries----------------

from PIL import Image
import numpy as  np
import cv2
from matplotlib import pyplot as plt
import subprocess as sub
import sys
import os
import math

##--------------Subroutines----------------

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


cv2.imwrite(new_image_path, image)
cv2.waitKey(0)
