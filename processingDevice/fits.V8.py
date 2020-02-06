# Libraries
import math
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import odr
from matplotlib.patches import Ellipse, Rectangle, Patch
import scipy as sp
#import csv

# Functions
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

    
######## MAIN #########

# Image Properties
coordsFile = 'output_coordinates.txt'
imageFile = 'tester_alterred.pgm'
global img
img = getImage(imageFile)
vLines = [0, 21, 33, 56, 70, 92]
targetFeatures = [ 1, 2, 3, 5, 6, 7, 8, 9,10, 11, 12, 13, 19] # dece ones
#targetFeatures = [ 1, 3, 5, 6, 8, 10, 11, 12, 13, 19] # prez ones
#targetFeatures = [ 15, 16, 17, 18, 20] # bad ones
#targetFeatures = [6,19] # awesome example ones
display = 0 # 1 For output printing, 0 for none.


descTable = mother(coordsFile, imageFile, vLines, targetFeatures, display)

for desc in descTable:
    print(desc)


plt.show()
######## END #########






