#!/usr/bin/python

import sys
import os
import dlib
import glob
from skimage import io
import cv2
from utils import *
import matplotlib.pyplot as plt
import numpy

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

print "Gathering images"
import glob
files = glob.glob("*.png")
files.sort()

fig = plt.figure()

rows = math.ceil(math.sqrt(len(files)))
cols = math.ceil(math.sqrt(len(files)))

for i,fn in enumerate(files):
    fig.add_subplot(rows,cols,i+1)
    img = cv2.imread(fn)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    try:
        points = get_landmarks(img)
        plt.title("distance: %.2f" % frontface_score(points))
    except:
        plt.title("no face detected")
        print "Got no landmarks..."
    

plt.show()
