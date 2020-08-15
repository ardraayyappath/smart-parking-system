import numpy as np
import cv2
#import imutils
#import sys
import pytesseract
import pandas as pd
#import time
#import glob
#import csv
import arrow
from imutils import contours
import os
print(os.getcwd())


#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# car object with attributes of vehicles in parking lot
class car:
    def __init__(self, lpn, entry, colour):
        self.entry=entry
        self.lpn=lpn
        self.colour=colour
current = 1
csv_path = 'colors.csv'
#path = "C:\\Users\\shant\\Documents\\Python Scripts\\car\\vehicle5.png"

# reading csv file
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv(csv_path, names=index, header=None)


#get colour name
def get_color_name(R,G,B):
	minimum = 1000
	for i in range(len(df)):
		d = abs(R - int(df.loc[i,'R'])) + abs(G - int(df.loc[i,'G'])) + abs(B - int(df.loc[i,'B']))
		if d <= minimum:
			minimum = d
			cname = df.loc[i, 'color_name']

	return cname


#get r,g,b value
def color_histogram_of_test_image(test_src_image):

    
# load the image
    crop_img = cv2.imread('car.jpeg')
    crop_img =  cv2.resize(crop_img,(400, 400))
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            
    r=int(red)
    g=int(green)
    b=int(blue)
    #print(r)
    text = get_color_name(r,g,b)
    print("colour name :", text)
    #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle
    cv2.rectangle(crop_img, (20,20), (600,60), (b,g,r), -1)
    #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType ) 
    cv2.putText(crop_img, text, (50,50), 2,0.8, (255,255,255),2,cv2.LINE_AA)
    cv2.imshow("img",crop_img)
    #print(feature_data)
    return text

    
#crop the center image
def crop_center(img,cropx,cropy): # to crop and get the center of the given image
    y,x, channels = img.shape
   
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)  
  
    return img[starty:starty+cropy,startx:startx+cropx]



def colour_detect():
    crop_img = cv2.imread('car4.jpeg')
    crop_img =  cv2.resize(crop_img,(400, 400))
#print(crop_img)
    #cv2.imshow("img1",crop_img)
    img=crop_center(crop_img, 70, 70)
#cv2.imshow("img3",img)
    c=color_histogram_of_test_image(img)
    #print(c, " colour ")
    return c




#pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
def get_license_plate_number():
    image = cv2.imread('car.jpeg')
    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    plate = ""
    lpn=""
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        center_y = y + h/2
        if area > 3000 and (w > h) and center_y > height/2:
        
            ROI = image[y:y+h, x:x+w]
            data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
            plate += data
            for ch in plate:
                if ch.isalnum():
                    lpn+=ch
                    
    #print("Detected LPN : ",lpn)
    return lpn
        

def get_entry_time():
    # return arrow.now()
    # use dummytime for now for better demonstration
    s = '2020-08-06 23:30:45'
    entry = arrow.get(s, 'YYYY-MM-DD HH:mm:ss')
    return entry


#calculates parking fees
def parking_fee(entry):
    exit = arrow.now()
    duration = exit - entry
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    total = hours * 60 + minutes
    fee = total * 3
    print("Total duration = ", hours, " : ", minutes)
    print("Parking Fee : ", fee)
    return fee

#checks if the given car is in the lot, outputs time and fees
def capture_position(platenum,cur):
    flag =0
    for key in spots:
        if spots[key].lpn == platenum:
            print("entry time : ", parking_fee(spots[key].entry))
            print("parking spot : ", key)
            flag =1
            cur-=1
            print("number of occupied spots (at time of exit) : ", cur)
            break
    if flag==0:
        print("could not find the car")
    return cur

tot_spots=100 # total number of spots, input from task 1
spots={} # dictionry format license plate : spot
spot_ID=[]
for k in range(100):
    spot_ID.append(k+1)
cars = list() # append when new car enters
tot_occupied=1
#for i in range(tot_occupied): # loop fpr demo
    #cars.append(car(get_license_plate_number(), get_entry_time()))
    #spots[spot_ID[i]] = cars[i]

cars.append(car('A1', get_entry_time(),'red'))
#dummy car in first spot
spots[spot_ID[1]] = cars[0]

#function to be called when a new car enters
def entry(cur):
    print("Welcome to smart parking system")
    c=car(get_license_plate_number(), get_entry_time(), colour_detect())
    cars.append(c)
    print("license plate number : ", c.lpn)
    print("entry time : ", c.entry)
    print("colour : ", c.colour)
    cur+=1
    spots[spot_ID[cur]] = c
    print("number of occupied spots (at time of entry) : ", cur)
    return cur

#function to be called when a car exits
def exit(cur):
    print("Enter license plate number of the exiting car ")
    capture_position(input(),cur)

print("printing dictionary of cars and parking spots")
for key,value in spots.items():
    print(key," : ", value.lpn)
current = entry(current)
current = exit(current)
#print(colour_detect())