#!/usr/bin/env python

##--------------Libraries----------------

from PIL import Image
import numpy as  np
from numpy import linalg as LA
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import Ellipse, Arc, Rectangle, Patch
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
from scipy import odr
import scipy as sp
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
                    print("Error")
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
                    print("Error")
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
            try:
                image[y1,x] = [255,0,0]
                image[y2,x] = [255,0,0]
            except IndexError:
                next

        y = min(y1,y2)
        while y < max(y1,y2):
            try:
                image[y,x1] = [255,0,0]
                image[y,x2] = [255,0,0]
                y += 1
            except IndexError:
                y += 1
                next

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

def addBox2(event, x, y, flags, params):
    global added_boxes

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (int(x),int(y))
            print("START: " + str((x,y)))
        if event == cv2.EVENT_LBUTTONUP:
            end = (int(x), int(y))
            print("END: " + str((x,y)))
        added_boxes.append(start,end)

def addBox(event, x, y, flags, param):
    global added_box_points, adding

    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        added_box_points.append(start)
        adding = True

    elif event == cv2.EVENT_LBUTTONUP:
        end = (x,y)
        added_box_points.append(end)
        adding = False

def removeBox(event, x, y, flags, params):
    global removed_boxes

    if event == cv2.EVENT_LBUTTONUP:
        click = (int(x), int(y))
        print("CLICK: " + str((x,y)))
        removed_boxes.append(click)



# OLD ELLIPSE FITTING

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

# NEW ELLIPSE FITTING

def getCoords(coordsFile):
    with open(coordsFile, 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)
    return processCoords(coords)

def processCoords(coords):
    procX = []
    procY = []
    for feat in coords:
        featX = [] 
        featY = []
        for point in feat:
            if point.startswith("["):
                point = point[2:-1]
            elif point.endswith("]"):
                point = point[2:-2]
            else:
                point = point[2:-1]
            point = complex(point)
            featX.append(point.real)
            featY.append(point.imag)
        featX = np.asarray(featX)
        featY = np.asarray(featY)
        procX.append(featX)
        procY.append(featY)
    return [procX, procY]

def getImage(imageFile):
    plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    img=mpimg.imread(imageFile)
    imgplot = plt.imshow(img, cmap='gray')
    fig = plt.gcf()
    ax = fig.gca()
    return img

def plotVlines(vLines):
    for v in vLines:
        plt.vlines(x=v, ymin=0, ymax = 569, color = 'xkcd:sky blue')
        return
        
def fitAll(xCoords, yCoords, vLines, target, disp):
    totalErr = 0
    descTable = []
    descTable.append(['xMid', 'yMid', 'alpha', 'Nbeta', 'feature No'])
    for i in range(0,len(xCoords)):
        xFeat = xCoords[i] 
        yFeat = yCoords[i]
        if i in target:
            print('Feature: ' + str(i))
            description = fitOne(xFeat,yFeat,vLines, disp, i)
            description.append(i)
            descTable.append(description)
    return descTable

def fitOne(xFeat, yFeat, vLines, disp, featNo):
    #plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
    imgplot = plt.imshow(img, cmap='gray')
    plt.scatter(xFeat, yFeat, s=12, c="purple", alpha=0.5) # C
    plt.title('Feature ' + str(featNo))
    bounds = whichCore(xFeat[0], vLines)

    #plt.scatter(xFeat[0], yFeat[0], s=10, c="pink", alpha=0.5)
    lbf = np.polyfit(xFeat, yFeat, 1)
    radAppx = getRadAppx(lbf,bounds[0],bounds[1])
    [chi, err] = ODRregress(xFeat, yFeat, radAppx, bounds, lbf[0])
    plt.scatter(chi[2], chi[3], s=8, c="k", alpha=0.5) # C
    peaks = np.asarray(getPeaks(chi))
    depth = getDepth(bounds,peaks[0])
    trueRad = getTrueRad(peaks, depth)
    growth = trueRad/max(chi[0:2])*100
    alpha = getAlpha(trueRad,bounds)
    beta = getBeta(peaks[0],bounds)
    [newBeta, quadrant] = correctBeta(xFeat,yFeat,peaks[0:2],peaks[2:4],beta)

    xMid = round(np.mean([peaks[0],peaks[2]]),2)
    yMid = round(np.mean([peaks[1],peaks[3]]),2)
    return [xMid, yMid, round(alpha,2), round(newBeta,2)]

def correctBeta(xFeat,yFeat, bottom, top, beta):
    point = [np.mean(xFeat),np.mean(yFeat)]
    #plt.scatter(point[0], point[1], s=10, c="pink", alpha=0.5) # C
    x = bottom[0]
    xMid = np.mean([x,top[0]])
    if beta == -1:
        return [-1, 'fail']
    elif lineSolve(point, bottom, top): # Back half.
        return [270 - beta, 'back']
    elif x < xMid: # Front left.
        return [(beta - 90) % 360, 'front left']
    else: # Front right.
        return [270 + beta % 360, 'front right']

def lineSolve(point, bottom, top):
    a = (top[1] - bottom[1])/(top[0] - bottom[0])
    b = bottom[1]- a*bottom[0]
    if point[1] > lineF(a,b,point[0]):
        return False
    else:
        return True

def lineF(a,b,x):
    return a*x + b
        
def plotEll(beta, color):
    # beta = [a, b, xc, yc, alpha] FYI: contents of beta.
    c = max(beta[0],beta[1])
    xlist = np.linspace(beta[2]-2*c, beta[2]+2*c, 1000)
    ylist = np.linspace(beta[3]-2*c, beta[3]+2*c, 1000) 
    X,Y = np.meshgrid(xlist, ylist)
    F = (( (X-beta[2])*math.cos(beta[4])+(Y-beta[3])*math.sin(beta[4])  )**2)/(beta[0]**2) \
        + (( (X-beta[2])*math.sin(beta[4]) - (Y-beta[3])*math.cos(beta[4]) )**2)/(beta[1]**2) - 1
    contourObj = plt.contour(X, Y, F, [0], colors = color, linestyles = 'solid')
    contourObj = contourObj.collections[0].get_paths()[0]
    return contourObj.vertices

def plotLine(a,b, x_range):  
    x = np.array(x_range)
    formula = str(a) + '*x + ' + str(b)
    y = eval(formula)
    plt.plot(x, y,'r')  

def getRadAppx(lbf,x1,x2):
    a = lbf[0]
    b = lbf[1]
    x1 = 33
    x2 = 55
    y1 = a*x1 + b
    y2 = a*x2 + b
    p1 = np.array([x1,y1])
    p2 = np.array([x2,y2])
    radAppx = np.linalg.norm(p1-p2)
    return(radAppx/2)

def whichCore(x,vLines):
    mids = [ np.mean(vLines[0:2]), np.mean(vLines[2:4]), np.mean(vLines[4:])]
    one = abs(x - mids[0])
    two = abs(x - mids[1])
    three = abs(x - mids[2])
    closest = min(one, two, three)
    if one == closest:
        return vLines[0:2]
    elif two == closest:
        return vLines[2:4]
    else: # 3 is closest
        return vLines[4:]

def ODRregress(xFeat, yFeat, radAppx, bounds, slope):
    xc = np.mean(bounds)
    yc = np.mean(yFeat)
    xx = np.array([xFeat, yFeat])
    mdr = odr.Model(f, implicit=True)
    mydata = odr.Data(xx, y=1)
    betaGuess =  [radAppx, 2, xc, yc, math.atan(slope)]
    #plotEll(betaGuess, 'y') # Initial guess. Yellow
    fixed =         [  0  , 1 , 0 , 1 , 1 ] # Holds some variables constant.
    myodr = odr.ODR(mydata, mdr, beta0 = betaGuess, ifixb = fixed)
    myoutput = myodr.run()
    return [myoutput.beta, myoutput.sum_square]

def f(B, x):
    return (( (x[0]-B[2])*math.cos(B[4])+(x[1]-B[3])*math.sin(B[4])  )**2)/(B[0]**2)  + (( (x[0]-B[2])*math.sin(B[4]) - (x[1]-B[3])*math.cos(B[4]) )**2)/(B[1]**2)-1.

def getPeaks(chi):
    verts = plotEll(chi,'g') # Fitted.
    peaks = getPeaksSub(verts)
    return peaks

def getPeaksSub(verts):
    xs = []
    ys = []
    xLow = 0
    yLow = 0
    xHigh = 700
    yHigh= 700
    for point in verts:
        xs.append(point[0])
        ys.append(point[1])
        if point[1] > yLow:
            xLow = point[0]
            yLow = point[1]
        if point[1] < yHigh:
            xHigh = point[0]
            yHigh = point[1]
            
    #plt.scatter(xLow, yLow, s=20, c="red", alpha=0.5)
    #plt.scatter(xHigh, yHigh, s=30, c="red", alpha=0.5)
    return [xLow, yLow, xHigh, yHigh]

def getDepth(bounds, xCoord):
    center = np.mean(bounds)
    radius = (bounds[1]-bounds[0])/2
    depth = (radius**2 - (xCoord-center)**2)**(1/2)
    return(depth)

def getTrueRad(peaks,depth):
    p1 = np.asarray([peaks[0], peaks[1],  depth]) # [x, y, z]
    p2 = np.asarray([peaks[2], peaks[3], -depth])
    return np.linalg.norm(p1-p2)/2

def getAlpha(trueRad, bounds):
    coreDiam = 1.0*bounds[1]-bounds[0]
    faceDiam = 2*trueRad
    try:
        alpha = math.asin(coreDiam/faceDiam)/math.pi*180
    except ValueError:
        alpha =-1
    return alpha   

def getBeta(botX, bounds):
    coreRad = (1.0*bounds[1]-bounds[0])/2
    print(coreRad)
    xDisplace = botX - np.mean(bounds)
    try:
        beta = math.acos(xDisplace/coreRad)/math.pi*180
    except ValueError:
        beta = -1
    return beta


def mother(coordsFile, imageFile, vLines, target, display):
    [xCoords, yCoords] = getCoords(coordsFile)
    #ax = getImage(imageFile)
    #plotVlines(vLines)
    descTable = fitAll(xCoords, yCoords, vLines, target, display)
    edges = Patch(color='purple', label='Cluster Edges')
    guessed = Patch(color='yellow', label='Initial Ellipse')
    fitted = Patch(color='green', label='Fitted Ellipse')
    bounds = Patch(color='blue', label='Cluster Bounds')
    plt.legend(handles=[edges,  fitted], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('x (Pixels)')
    plt.ylabel('y (Pixels)')
    return descTable



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

def removeSmallBoxes(boxes, image, xvals):
    new_boxes = []
    for i,box in enumerate(boxes):
        count = 0
        big_enough = False
        y = box[0]
        while y <= box[1]:
            for x in xvals:
                if image[y,x][0] == 0 and image[y,x][1] == 255 and image[y,x][2] == 0:
                    count = count + 1
            if count > MIN_CLUSTER_POINTS:
                big_enough = True
                break
            y = y + 1
        if big_enough:
            new_boxes.append(box)
    return new_boxes

MIN_LINE_LENGTH = 3
MAX_SLOPE = 5
CORE_WIDTH = 31 
MAX_DISTANCE = 8
CORE1_XVALS = range(1,31)
CORE2_XVALS = range(32,62)
CORE3_XVALS = range(63,94)
NUM_CLUSTERS = 20
MIN_CLUSTER_POINTS = 8
SNAP_DIST = 9

removed_boxes = []
added_box_points = []
added_boxes = []
adding = False
##--------------Program----------------

# Copy to a new image so the original doesn't get alterred

image_path = sys.argv[1]
new_image_path = os.path.splitext(image_path)[0]+"_alterred"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, new_image_path])

# For later when we take user input 
image_with_deleted_boxes = os.path.splitext(image_path)[0]+"_with_boxes_removed"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, image_with_deleted_boxes])
im_w_del = cv2.imread(image_with_deleted_boxes)

image_with_added_boxes = os.path.splitext(image_path)[0]+"_with_boxes_added"+os.path.splitext(image_path)[1]
sub.call(["cp", "-f", image_path, image_with_added_boxes])
im_w_add = cv2.imread(image_with_added_boxes)

# Read in new image for editing

image = cv2.imread(new_image_path)

# Apply original ELSDc with no modification to parameters

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_image_path = os.path.splitext(new_image_path)[0]+".pgm"
cv2.imwrite(grey_image_path, grey_image)
home_dir = os.path.expanduser("~")
elsdc_path = os.path.dirname(sys.argv[0]) + "/../elsdc/src/elsdc"
cmd = elsdc_path + " " + grey_image_path
os.system(cmd)

# Convert output.svg to a txt file containing X,Y coordinates for all lines

svgToCoords(os.path.dirname(new_image_path)+"/output.svg")

# Draw all detected lines in a given colour

coords_file = os.path.dirname(new_image_path)+"/output_coordinates.txt"

cf = open(coords_file, 'r')

for i, line in enumerate(cf.readlines()):
    drawLine(line.rstrip("\n"), image)
    drawLine(line.rstrip("\n"), im_w_del)
    drawLine(line.rstrip("\n"), im_w_add)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows


cf.close()

# Scan all 3 cores for areas of high density and 'box' them

core1_file = os.path.dirname(new_image_path)+"/output_core1.txt"
core2_file = os.path.dirname(new_image_path)+"/output_core2.txt"
core3_file = os.path.dirname(new_image_path)+"/output_core3.txt"

height = image.shape[0]

for i in [1, 2, 3]:
    if i == 1:
        f = open(core1_file, 'r')
        core1boxes = []
        xvals = CORE1_XVALS
    elif i == 2:
        f = open(core2_file, 'r')
        core2boxes = []
        xvals = CORE2_XVALS
    else:
        f = open(core3_file, 'r')
        core3boxes = []
        xvals = CORE3_XVALS

    data = []
    for line in f.readlines():
        line.rstrip(',\n')
        for point in line.split(","): 
            if point != '\n':
                data.append((complex(point).real,complex(point).imag))
    
    (mu, clusters) = find_centers(data, NUM_CLUSTERS)
    colourClusters(mu, image)
    colourClusters(mu, im_w_del)
    colourClusters(mu, im_w_add)

    boxes = []

    for centroid in enumerate(mu):
        try:
            cluster_length = getClusterLength(clusters[centroid[0]])
            boxes.append([int(centroid[1][1]-int(cluster_length/2)), int(centroid[1][1]+int(cluster_length/2))])
        except KeyError:
            continue
    #box_file = os.path.dirname(new_image_path)+"/core" + str(i) + "_boxes.txt"
    if boxes != []:
        boxes = mergeBoxes(boxes)
        boxes = removeSmallBoxes(boxes,image,xvals)
        if i == 1:
            core1boxes = boxes
        elif i == 2:
            core2boxes = boxes
        else:
            core3boxes = boxes
        #saveBoxes(boxes, box_file)
        #colourBoxes(boxes, im_w_del, xvals, i)
        #colourBoxes(boxes, im_w_add, xvals, i)
        cv2.imwrite(new_image_path, image)
    f.close()

colourBoxes(core1boxes, image, CORE1_XVALS, 1)
colourBoxes(core2boxes, image, CORE2_XVALS, 2)
colourBoxes(core3boxes, image, CORE3_XVALS, 3)
cv2.imwrite(new_image_path, image)

cv2.namedWindow('Click Any Boxes to Remove', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Click Any Boxes to Remove', removeBox)
cv2.imshow('Click Any Boxes to Remove', image)
cv2.waitKey(0)
cv2.destroyAllWindows

core1_to_remove = []
core2_to_remove = []
core3_to_remove = []

for click in removed_boxes:
    if click[0] in CORE1_XVALS:
        for i,box in enumerate(core1boxes):
            if click[1] in range(box[0],box[1]):
                core1_to_remove.append(i)
    elif click[0] in CORE2_XVALS:
        for i,box in enumerate(core2boxes):
            if click[1] in range(box[0],box[1]):
                core2_to_remove.append(i)
    elif click[0] in CORE3_XVALS:
        for i,box in enumerate(core3boxes):
            if click[1] in range(box[0],box[1]):
                core3_to_remove.append(i)
    else:
        print("Click out of range, skipping this box deletion")

#print "CORE 1 TO REMOVE" + str(core1_to_remove)
core1_to_remove.sort(reverse=True)
for i in core1_to_remove:
    del core1boxes[i]

core2_to_remove.sort(reverse=True)
for i in core2_to_remove:
    del core2boxes[i]

core3_to_remove.sort(reverse=True)
for i in core3_to_remove:
    del core3boxes[i]

colourBoxes(core1boxes, im_w_del, CORE1_XVALS, 1)
colourBoxes(core2boxes, im_w_del, CORE2_XVALS, 2)
colourBoxes(core3boxes, im_w_del, CORE3_XVALS, 3)

cv2.imwrite(image_with_deleted_boxes, im_w_del)

cv2.namedWindow('Click and Drag to Create New  Boxes', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Click and Drag to Create New  Boxes', addBox)
cv2.imshow('Click and Drag to Create New  Boxes', im_w_del)
cv2.waitKey(0)
cv2.destroyAllWindows

box = [(1,2),(3,4)]
for i,point in enumerate(added_box_points):
    if i%2 == 0:
        box[0] = point
    else:
        box[1] = point
        if box[0][1] > box[1][1]:
            tmp = box[0]
            box[0] = box[1]
            box[1] = tmp
        added_boxes.append(box)

for box in added_boxes:
    if box[0][0] in CORE1_XVALS and box[1][0] in CORE1_XVALS:
        core1boxes.append((box[0][1],box[1][1]))
    elif box[0][0] in CORE2_XVALS and box[1][0] in CORE2_XVALS:
        core2boxes.append((box[0][1],box[1][1]))
    elif box[0][0] in CORE3_XVALS and box[1][0] in CORE3_XVALS:
        core3boxes.append((box[0][1],box[1][1]))
    else:
        print("Box not included, xvals not in range")

colourBoxes(core1boxes, im_w_add, CORE1_XVALS, 1)
colourBoxes(core2boxes, im_w_add, CORE2_XVALS, 2)
colourBoxes(core3boxes, im_w_add, CORE3_XVALS, 3)

print(len(core1boxes) + len(core2boxes) + len(core3boxes))

cv2.imwrite(image_with_added_boxes, im_w_add)

cv2.namedWindow('This image will be used to fit ellipses', cv2.WINDOW_NORMAL)
cv2.imshow('This image will be used to fit ellipses', im_w_add)
cv2.waitKey(0)
cv2.destroyAllWindows

imageFile = os.path.splitext(image_path)[0]+"_alterred.pgm"
global img
img = getImage(imageFile)
vLines = [0, 21, 33, 56, 70, 92]
targetFeatures = [ 1, 2, 3, 5, 6, 7, 8, 9,10, 11, 12, 13, 19] 
#targetFeatures = [ 1, 3, 5, 6, 8, 10, 11, 12, 13, 19] # prez ones
#targetFeatures = [ 15, 16, 17, 18, 20] # bad ones
#targetFeatures = [6,19] # awesome example ones
display = 0 # 1 For output printing, 0 for none.


descTable = mother(coords_file, imageFile, vLines, targetFeatures, display)

for desc in descTable:
    print(desc)


plt.show()

#cv2.imwrite(new_image_path, image)


