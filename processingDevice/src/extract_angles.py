import numpy as np


## raw dummy coordinates (same format as output_coordinates.txt)
left_line = [ (1+1j), (1+2j), (1+3j), (1+4j), (1+5j) ] 
right_line = [ (6+1j), (6+2j), (6+3j), (6+4j), (6+5j) ]
ellipse = [ (1+5j), (2+4j), (3+3j), (3.5+3j),(4+3.5j), (6+4j)]


ellipse =np.array(ellipse) 
peak = ellipse[ellipse.imag.argmin()] ##find lowest peak (trough) of ellipse

radius = abs(np.array(right_line[1]).real - np.array(left_line[1]).real)/float(2) ##radius of core

ellipse_center = np.median(ellipse.imag) #this is defitely off since I'm uncertain about what the center point means

alpha = np.arctan( abs(ellipse_center - peak.imag)/float(radius) ) #alpha angle

beta = np.arcsin( abs(radius - peak.real)/float(radius) ) #beta angle

print "Alpha angle: " + str(alpha)
print "Beta angle: " + str(beta)





	








