
from PIL import Image
import numpy as  np
import cv2
from matplotlib import pyplot as plt
import subprocess as sub
import sys
import os
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import ellipse_perimeter
from skimage.util import img_as_ubyte

# Copy to a new image so the original doesn't get alterred
image_path = sys.argv[1]
new_image_path = os.path.splitext(image_path)[0]+"_alterred"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, new_image_path])

# Change pixel values
ih = Image.open(new_image_path)
pic = ih.load()
i = 0
j = 0
while (i < 960):
	while (j < 540):
		pic[i, j] = (0, 240, 0)
		j += 1
	i += 1
ih.show()

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

# Hough transform
result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
result.sort(order='accumulator')

best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
