# importing libraries
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract


# Reading image from folder where it is stored
img = cv2.imread('test5.jpeg')
cv2.imshow('image-original',img)
cv2.waitKey(0)

# denoising of image saving it into dst image
#works with a color image.
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

# Plotting of source and destination image
cv2.imshow('image-denoised',dst)
cv2.waitKey(0)
cv2.imwrite('denoised5.png',dst)
print("denoised")

im = Image.open("denoised5.png")


text = pytesseract.image_to_string(im, lang = 'eng')
#print("read")

print(text)

