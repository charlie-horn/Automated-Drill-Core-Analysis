
from PIL import Image
import numpy as  np
import cv2
from matplotlib import pyplot as plt
import subprocess as sub
import sys
import os



# Copy to a new image so the original doesn't get alterred
image_path = sys.argv[1]
new_image_path = os.path.splitext(image_path)[0]+"_alterred"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, new_image_path])

# Get click event pixel value

def click(event, x, y, flags, params):
	global points

	if event == cv2.EVENT_LBUTTONUP:
		points.append((x, y))
		RGB = pixels[x,y]
		print (x,y)
		print RGB
		rgbs.append(RGB)

image = cv2.imread(new_image_path)
ih = Image.open(new_image_path)
pixels = ih.load()
points = []
rgbs = []
clicking = False
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

while True:
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		cv2.destroyAllWindows()
		break

# Change pixel values
Rsum = 0
Gsum = 0
Bsum = 0

for rgb in rgbs:
	Rsum += rgb[0]
	Gsum += rgb[1]
	Bsum += rgb[2]

Rmean = Rsum/len(rgbs)
print "Rmean : " + str(Rmean)
Gmean = Gsum/len(rgbs)
print "Gmean : " + str(Gmean)
Bmean = Bsum/len(rgbs)
print "Bmean : " + str(Bmean)

#for i in range(ih.size[0]):
#	for j in range(ih.size[1]):
#		if pixel_fits_criteria(pixel, Rmean, Gmean, Bmean):
#			pixels[i,j] = (240,240,240)
#	ih.show()

# Display original and edge
img = cv2.imread(new_image_path)
edges = cv2.Canny(img,100,200)

plt.subplot(311)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])
plt.subplot(312)
plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()

# ELSDc 

grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey_image_path = os.path.splitext(image_path)[0]+"_alterred.pgm"
cv2.imwrite(grey_image_path, grey_image)
sub.call(["/home/rockmass/mthe494/catkin_ws/src/elsdc/src/elsdc", grey_image_path])

# Find longest vertical polygons from output
currentDir = os.getcwd()
polygons = currentDir + "out_polygon.txt"
for polygon in polygons:
	split_line = polygon.split(' ')
	index = split_line[0]
	length = 




