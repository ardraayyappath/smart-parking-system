import cv2
import numpy as np
import pytesseract

#-----Reading the image-----------------------------------------------------
img = cv2.imread('test.jpeg', 1)
#cv2.imshow("img",img)
dst = cv2.fastNlMeansDenoisingColored(img, None, 10 , 12, 7, 15)

#-----Converting image to LAB Color model-----------------------------------
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
#cv2.imshow('l_channel', l)
#cv2.imshow('a_channel', a)
#cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)


#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))


#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


#text1 = pytesseract.image_to_string(final, lang = 'eng')
#text2 = pytesseract.image_to_string(limg, lang = 'eng')
#text3 = pytesseract.image_to_string(cl, lang = 'eng')
#text4 = pytesseract.image_to_string(lab, lang = 'eng')
dst1 = cv2.fastNlMeansDenoisingColored(lab, None, 10 , 10, 7, 15)
#dst2 = cv2.fastNlMeansDenoisingColored(cl, None, 10 , 10, 7, 15)
dst3 = cv2.fastNlMeansDenoisingColored(limg, None, 10 , 10, 7, 15)
dst4 = cv2.fastNlMeansDenoisingColored(final, None, 10 , 12, 7, 14)
#cv2.imshow('final-denoised', dst)
#cv2.waitKey(0)
text1 = pytesseract.image_to_string(dst1, lang = 'eng')
#text2 = pytesseract.image_to_string(dst2, lang = 'eng')
text3 = pytesseract.image_to_string(dst3, lang = 'eng')
text4 = pytesseract.image_to_string(dst4, lang = 'eng')
text5 = pytesseract.image_to_string(dst, lang = 'eng')


print("first",text1)
#print("second",text2)
print("third",text3)
print("fourth",text4)
print("fifth",text5)
