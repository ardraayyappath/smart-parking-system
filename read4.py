import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
img2=cv2.imread('test.jpeg')
dst = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 15)

img = cv2.imread('test.jpeg',0)

text = pytesseract.image_to_string(dst, lang='eng')
print("text with simple denoising : ", text)
#print(text)

img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,4)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]


for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

    text = pytesseract.image_to_string(images[i], lang='eng')
    print("text with filter :", i + 1, " ", text)

plt.show()