

##--------------Libraries----------------

from PIL import Image
import numpy as  np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import Ellipse, Arc, Rectangle
from numpy.linalg import eig, inv
import subprocess as sub
import sys
import csv
import os
import math
import re
import shutil
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier
import random
##--------------Subroutines----------------

def svgToCoords(svg_path):
	trunc = os.path.splitext(svg_path)[0]
	txt_path = trunc + ".txt"

	shutil.copyfile(svg_path, txt_path)

	in_path = txt_path
	out_path = trunc + "_coordinates.txt"
	core_1_path = trunc + "_core1.txt"
	core_2_path = trunc + "_core2.txt"
	core_3_path = trunc + "_core3.txt"
	
	in_f = open(in_path, 'r')
	out_f = open(out_path, 'w')
	core_1 = open(core_1_path, 'w')
	core_2 = open(core_2_path, 'w')
	core_3 = open(core_3_path, 'w')
	
	f = in_f.readlines()

	line_pattern = re.compile(r'<(?P<type>\w{4}) x1=\"(?P<x1>-?\d+\.?\d*)\" y1=\"(?P<y1>-?\d+\.?\d*)\" x2=\"(?P<x2>-?\d+\.\d*?)\" y2=\"(?P<y2>-?\d+\.?\d*)\".*')
	path_pattern = re.compile(r'<(?P<type>\w{4}) d=\"M (?P<x1>-?\d+\.?\d*),(?P<y1>-?\d+\.?\d*) A(?P<r1>\d+\.?\d*),(?P<r2>\d+\.?\d*) (?P<rotation>-?\d+\.?\d*) (?P<arc>\d{1}),(?P<sweep>\d{1}) (?P<x2>-?\d+\.?\d*),(?P<y2>-?\d+\.?\d*)\".*')
	
	for i, line in enumerate(f):

		if line.startswith("<line"):
			
			line_match = re.match(line_pattern, line)

			x1 = float(line_match.group('x1'))
			y1 = float(line_match.group('y1'))
			x2 = float(line_match.group('x2'))
			y2 = float(line_match.group('y2'))

			start = complex(x1,y1)
			middle = complex(int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
			end = complex(float(x2),float(y2))

			coords = getLineCoords(start, end)
			if coords != []:
				if middle.real in CORE1_XVALS:
					for i in coords: core_1.write(str(i)+",")
					core_1.write("\n")
				elif middle.real in CORE2_XVALS:
					for i in coords: core_2.write(str(i)+",")
					core_2.write("\n")
				elif middle.real in CORE3_XVALS:
					for i in coords: core_3.write(str(i)+",")
					core_3.write("\n")
				else:
					print "SOMETHING FUCKED UP"
				out_f.write(str(coords))
				out_f.write("\n")
			
		elif line.startswith("<path"):

			path_match = re.match(path_pattern, line)
			x1 = float(path_match.group('x1'))
			y1 = float(path_match.group('y1'))
			r1 = path_match.group('r1')
			r2 = path_match.group('r2')
			rotation = float(path_match.group('rotation'))
			arc = int(path_match.group('arc'))
			sweep = int(path_match.group('sweep'))
			x2 = float(path_match.group('x2'))
			y2 = float(path_match.group('y2'))

			start = complex(x1,y1)
			radius = complex(float(r1),float(r2))
			middle = complex(int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
			end = complex(x2,y2)

			coords = getPathCoords(start, radius, rotation, arc, sweep, end)
			if coords != []:
				if middle.real in CORE1_XVALS:
					for i in coords: core_1.write(str(i)+",")
					core_1.write("\n")
				elif middle.real in CORE2_XVALS:
					for i in coords: core_2.write(str(i)+",")
					core_2.write("\n")
				elif middle.real in CORE3_XVALS:
					for i in coords: core_3.write(str(i)+",")
					core_3.write("\n")
				else:
					print "SOMETHING FUCKED UP"
				out_f.write(str(coords))
				out_f.write("\n")

	in_f.close()
	core_1.close()
	core_2.close()
	core_3.close()
	out_f.close()

def getLineCoords(start, end):
	path = Line(start, end)
	coords = []

	if path.length() < MIN_LINE_LENGTH:
		return coords

	slope = getSlope(start, end)
	if abs(slope) > MAX_SLOPE:
		return coords

	for i in drange(0, 1, 0.1):
		coords.append(path.point(i))
	return coords

def getPathCoords(start, radius, rotation, arc, sweep, end):
	path = Arc(start, radius, rotation, arc, sweep, end)
	coords = []

	if path.length() < MIN_LINE_LENGTH:
		return coords

	slope = getSlope(start, end)
	if abs(slope) > MAX_SLOPE:
		return coords

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

# CLUSTER

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
		minDist = None
		bestmukey = None
		for i in enumerate(mu):
			dist = np.linalg.norm([x[0]-i[1][0],x[1]-i[1][1]])
			if minDist is None or minDist > dist:
				minDist = dist
				bestmukey = i[0]
			else:
				continue
		try:
			clusters[bestmukey].append(x)
		except KeyError:
			clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    clusters = cluster_points(X, oldmu)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def getClusterLength(points):
	highest = None
	lowest = None
	for point in points:
		if highest is None:
			highest = point[1]
		elif point[1] > highest:
			highest = point[1]
		else:
			highest = highest

		if lowest is None:
			lowest = point[1]
		elif point[1] < lowest:
			lowest = point[1]
		else: 
			lowest = lowest

	distance = highest - lowest
	return distance

def colourBoxes(boxes, image, xvals, core):

	for box in boxes:
		x1 = xvals[0]
		y1 = box[1]
		x2 = xvals[-1]
		y2 = box[0]
		for x in xvals:	
			image[y1,x] = [255,0,0]
			image[y2,x] = [255,0,0]
	
		y = min(y1,y2)
		while y < max(y1,y2):
			image[y,x1] = [255,0,0]
			image[y,x2] = [255,0,0]
			y += 1

def colourClusters(clusters, image):
	for cluster in clusters:
		image[cluster[1],cluster[0]] = [0,0,255]
		image[cluster[1]+1,cluster[0]] = [0,0,255]
		image[cluster[1],cluster[0]+1] = [0,0,255]
		image[cluster[1],cluster[0]-1] = [0,0,255]
		image[cluster[1]-1,cluster[0]] = [0,0,255]

def saveBoxes(boxes, box_file):
	bf = open(box_file, 'w')
	for box in boxes:
		bf.write(str(box))
	bf.close()

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

# ELLIPSE FITTING

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else: 
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def fitToBox(boxes, image_path, f, xvals):
	
	img=mpimg.imread(image_path)
	imgplot = plt.imshow(img, cmap='gray')

	fig = plt.gcf()
	ax = fig.gca()

	for box in boxes:
		top = int(box[1])
		bot = int(box[0])
		left = xvals[0]
		right = xvals[-1]

		rec = Rectangle(
				(left, top),   # (x,y)
				right-left,          # width
				bot-top,          # height
				fill = False,
				edgecolor="red"
			)
		ax.add_patch(rec)

		x = []
		y = []
		xt = []
		yt = []

		f.seek(0)
		for line in f.readlines():
			coords = []
			line.rstrip(',\n')
			for point in line.split(","):
				if point != '\n':
					coords.append(complex(point))
			for point in coords:
				x.append(point.real)
				y.append(point.imag)
				if point.real > left and point.real < right:
					if point.imag > bot and point.imag < top:
						xt.append(point.real)
						yt.append(point.imag)
		if xt == [] or yt == []: continue
		plt.scatter(x, y, s=1, c="green", alpha=0.5)
		plt.scatter(xt, yt, s=1, c="blue", alpha=0.5)
		xt = np.asarray(xt)
		yt = np.asarray(yt)

		el = fitEllipse(xt,yt)
		center = ellipse_center(el)
		phi = ellipse_angle_of_rotation(el)
		axes = ellipse_axis_length(el)

		elFit = Ellipse(xy=(center[0], center[1]), width=2*axes[1], height=2*axes[0], angle=phi*180/math.pi, edgecolor='y', fc='None', lw=1)

		ax.add_patch(elFit)


		plt.show()

def mergeBoxes(boxes):
	changes = False
	new_boxes = []
	to_remove = []
	for i,ref_box in enumerate(boxes):
		if i in to_remove: next
		for j,comp_box in enumerate(boxes):
			if i == j: next
			if ref_box[0] < comp_box[1]+SNAP_DIST and ref_box[0] > comp_box[0]:
				changes = True
				ref_box[0] = comp_box[0]
				to_remove.append(j)
			elif ref_box[1] > comp_box[0]-SNAP_DIST and ref_box[1] < comp_box[1]:
				changes = True
				ref_box[1] = comp_box[1]
				to_remove.append(j)
			else:
				pass
		new_boxes.append(ref_box)
	if changes:
		return mergeBoxes(new_boxes)
	else:
		return new_boxes
##--------------Classes----------------

# class Feature {}

# class Line(Feature) {}

# class Ellipse(Feature) {}


##----------Global Variables-----------

MIN_LINE_LENGTH = 3
MAX_SLOPE = 5
CORE_WIDTH = 31 #TODO find the pixel width of core when we take a pic with the orbec
MAX_DISTANCE = 8
CORE1_XVALS = range(1,31)
CORE2_XVALS = range(32,62)
CORE3_XVALS = range(63,94)
NUM_CLUSTERS = 20
MIN_CLUSTER_POINTS = 5
SNAP_DIST = 7
##--------------Program----------------

# Copy to a new image so the original doesn't get alterred

image_path = sys.argv[1]
new_image_path = os.path.splitext(image_path)[0]+"_alterred"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, new_image_path])

# Read in new image for editing

image = cv2.imread(new_image_path)
#height, width = image.shape[:2]

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

cf.close()

# Scan all 3 cores for areas of high density and 'box' them

core1_file = os.path.dirname(new_image_path)+"/output_core1.txt"
core2_file = os.path.dirname(new_image_path)+"/output_core2.txt"
core3_file = os.path.dirname(new_image_path)+"/output_core3.txt"

height = image.shape[0]

for i in [1, 2, 3]:
	if i == 1:
		f = open(core1_file, 'r')
		xvals = CORE1_XVALS
	elif i == 2:
		f = open(core2_file, 'r')
		xvals = CORE2_XVALS
	else:
		f = open(core3_file, 'r')
		xvals = CORE3_XVALS

	data = []
	for line in f.readlines():
		line.rstrip(',\n')
		for point in line.split(","): 
			if point != '\n':
				data.append((complex(point).real,complex(point).imag))
	
	(mu, clusters) = find_centers(data, NUM_CLUSTERS)
	colourClusters(mu, image)

	boxes = []

	for centroid in enumerate(mu):
		try:
			if len(clusters[centroid[0]]) < MIN_CLUSTER_POINTS:
				next
			else:
				cluster_length = getClusterLength(clusters[centroid[0]])
				boxes.append([centroid[1][1]-int(cluster_length/2), centroid[1][1]+int(cluster_length/2)])
		except KeyError:
			continue
	box_file = os.path.dirname(new_image_path)+"/core" + str(i) + "_boxes.txt"
	if boxes != []:
		boxes = mergeBoxes(boxes)
		saveBoxes(boxes, box_file)
		colourBoxes(boxes, image, xvals, i)
		cv2.imwrite(new_image_path, image)
		#fitToBox(boxes, new_image_path, f, xvals)
	f.close()

# Write alterred pixels to the image
# TODO regularly write this and display to the user

	cv2.imwrite(new_image_path, image)


