import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from numpy.linalg import eig, inv
from PIL import Image
from matplotlib.patches import Ellipse, Arc, Rectangle
import csv

# This code takes the output_coordinates file, and the original pgm file. It then:
# 1. displays all detected edge points from output_coordinates.txt on the image
# 2. given a (currently hardcoded) rectangle of focus, code isolates points that fall within that rectangle
# 3. an ellipse is fit to the points that fall within the rectangle.

with open('output_coordinates.txt', 'r') as f:
  reader = csv.reader(f)
  coords = list(reader)

img=mpimg.imread('tester_alterred.pgm')
imgplot = plt.imshow(img, cmap='gray')

fig = plt.gcf()
ax = fig.gca()

# Rectangle
top = 95
bot = 122
left = 45
right = 55

rec = Rectangle(
        (left, top),   # (x,y)
        right-left,          # width
        bot-top,          # height
        fill = False,
        edgecolor="red"
    )
ax.add_patch(rec)

# Items
x = []
y = []
xt = []
yt = []

for feat in coords:
    for point in feat:
        if point.startswith("["):
            point = point[2:-1]
        elif point.endswith("]"):
            point = point[2:-2]
        else:
            point = point[2:-1]
        point = complex(point)
        x.append(point.real)
        y.append(point.imag)
        if point.real > left and point.real < right:
            if point.imag < bot and point.imag > top:
                xt.append(point.real)
                yt.append(point.imag)

plt.scatter(x, y, s=1, c="green", alpha=0.5)
plt.scatter(xt, yt, s=1, c="blue", alpha=0.5)


xt = np.asarray(xt)
yt = np.asarray(yt)


# Ellipse Fitting
# based on: http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

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

# Fit and map ellipse
el = fitEllipse(xt,yt)
center = ellipse_center(el)
phi = ellipse_angle_of_rotation(el)
#phi = ellipse_angle_of_rotation2(el)
axes = ellipse_axis_length(el)

elFit = Ellipse(xy=(center[0], center[1]), width=2*axes[1], height=2*axes[0], angle=phi*180/math.pi, edgecolor='y', fc='None', lw=1)

ax.add_patch(elFit)


plt.show()


