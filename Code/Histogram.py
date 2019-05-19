
#----------------------------Libraries---------------------------------#
import numpy as np
from numpy import array as ar
import cv2
from matplotlib import pyplot as plt

#------------------------Function Definition---------------------------#

# Function 1

def hand_contour_find(contours):
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv2.arcLength(cont,0)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        h_contour=contours[largest_contour]
        return True,h_contour

# Function 2

def histogram_calc(image):

    #selecting the pixels in the rectangles
    square1 = image[125:150,275:300]
    square2 = image[175:200,275:300]
    square3 = image[225:250,275:300]
    square4 = image[125:150,325:350]
    square5 = image[175:200,325:350]
    square6 = image[225:250,325:350]

    #Creating ROI subset
    sq=np.concatenate((square1,square2,square3,square4,square5,square6))

    # Histogram Calculation
    histo = cv2.calcHist([sq],[0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(histo, histo, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('Histogram',histo)
    return histo

#Function 3

##def back_proj(image,histo)
    back_projection = cv2.calcBackProject([image],[0,1],histo,[00,180,0,256],1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(back_projection,-1,disc,back_projection)
    ret, thresh = cv2.threshold(back_projection, 50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    cv2.imshow('Back Projection', thresh)
    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilate)
    return thresh
    
#---------------------------Program------------------------------------#
cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

ret,nittu = cap.read()
bg = cv2.createBackgroundSubtractorMOG2()
varun = cv2.cvtColor(nittu,cv2.COLOR_BGR2GRAY)

# get image properties.
h,w = np.shape(varun)
 
# print image properties.
#print "width: " + str(w)
#print "height: " + str(h)
cv2.imshow('varun', varun)
while(True):
    # Capture frame-by-frame                        
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tittu = cv2.medianBlur(frame,13)

    # Rectangles on left column
    cv2.rectangle(frame,(275,125),(300,150),(0,255,0),1)
    cv2.rectangle(frame,(275,175),(300,200),(0,255,0),1)
    cv2.rectangle(frame,(275,225),(300,250),(0,255,0),1)
    # Rectangles on right column
    cv2.rectangle(frame,(325,125),(350,150),(0,255,0),1)
    cv2.rectangle(frame,(325,175),(350,200),(0,255,0),1)
    cv2.rectangle(frame,(325,225),(350,250),(0,255,0),1)
    
    
    #BG Mask Apply
    fgmask = bg.apply(hsv)
##
##    #Gaussian Blur
    blur = cv2.medianBlur(fgmask,3)
##
##    #Thresholding
    ret, thresh_img = cv2.threshold(blur,127,255,cv2.THRESH_BINARY,1)
##
##   #Find Contour
    _, contours,hierarchy  = cv2.findContours(thresh_img ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found,hand_contour=hand_contour_find(contours)
    if(found):
        #Draw Contours
        cv2.drawContours(frame,hand_contour,-1,(0,255,0),2)
        #Hand Hull and Defects
        hand_convex_hull=cv2.convexHull(hand_contour,returnPoints = False)
        defects = cv2.convexityDefects(hand_contour,hand_convex_hull)

    # Display the resulting frame
    cv2.imshow('frame2',frame)
    cv2.imshow('frame3',hsv)
    cv2.imshow('frame4' ,tittu)

    if cv2.waitKey(1) & 0xFF == ord('h'):
        histogram_calc(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
