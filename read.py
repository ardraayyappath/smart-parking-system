import cv2
import numpy as np
import pytesseract

img = cv2.imread('test.jpeg')
dst = cv2.fastNlMeansDenoisingColored(img, None, 10 , 12, 6, 14)
#plate_image = cv2.convertScaleAbs(img, alpha=(2.0))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (1, 1), 0)
# convert to grayscale and blur the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
# Applied inversed thresh_binary
binary = cv2.threshold(blur, 80, 55,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
text1 = pytesseract.image_to_string(dst, lang = 'eng')
text2 = pytesseract.image_to_string(gray, lang = 'eng')
text3 = pytesseract.image_to_string(blur, lang = 'eng')
text4 = pytesseract.image_to_string(binary, lang = 'eng')


print("text after simple denoising : ", text1)
print("text after graying : ", text2)
print("text after gaussian blur : ", text3)
print("text after binary filter : ", text4)

