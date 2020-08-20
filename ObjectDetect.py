import cv2 as cv
import imutils as im
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def inside(r1,r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1 + w1 < (x2 + w2)) and (y1 + h1 < (y2 + h2))


def wrap_digit(rect, img_w, img_h):
    x, y, w, h = rect
    x_center = x + w//2
    y_center = y + h//2

    if (h > w):
        w = h
        x = x_center - (w//2)
    else:
        h = w
        y = y_center - (h//2)

    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    if x < 0:
        x = 0
    elif x > img_w:
        x = img_w
    if y < 0:
        y = 0
    elif y > img_h:
        y = img_h
    if y+h > img_h:
        h = img_h - y
    return x, y, w, h


#starts our
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))
clf = svm.SVC(gamma=.001)


#traing dataset
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=1, shuffle=False)
clf.fit(X_train,y_train)


#image manipulation
kernel = np.ones((2,2),np.uint8)
imageSource = cv.imread("Test3.jpg")
resized = im.resize(imageSource, width = 600)
image = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
#image = cv.GaussianBlur(image, (11,11),0)
ret, thresh = cv.threshold(image,127,255,cv.THRESH_BINARY_INV)
#thresh = cv.erode(thresh,kernel, thresh)
cnts, heir = cv.findContours(thresh.copy(),cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


img_h, img_w = resized.shape[:2]
img_area = img_h * img_w

#stores our valid numbers
rectangles = []


#finds our valid numbers

for c in cnts:
    a = cv.contourArea(c)
    if a >= .98 * img_area or a <= .0001 * img_area:
        continue
    r = cv.boundingRect(c)
    is_inside = False
    for q in rectangles:
        if inside(r,q):
            is_inside = True
            break
    if not is_inside:
        rectangles.append(r)
numb = 0
for r in rectangles:
    x, y, w, h = wrap_digit(r, img_w, img_h)
    roi = image[y:y+h, x:x+w]
    roi = cv.resize(roi,(64,64))
    cv.imshow("roi %d"%(numb),roi)
    numb += 1
    num = clf.predict(roi)
    print(roi)
    cv.rectangle(resized,(x,y),(x+w, y+h), (0, 255, 0),2)
    #cv.putText(resized,"%d" % (num), (x, y-5),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #print(num)
cv.imshow('img',resized)
cv.imshow('thresh',thresh)

cv.waitKey(0)



