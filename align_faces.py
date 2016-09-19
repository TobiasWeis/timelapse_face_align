#!/usr/bin/python

import cv2
import dlib
import numpy as np

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import glob

from settings import *
from utils import *


cascPath = "./haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

s = Settings()
net = s.net
net.load_weights_from(s.net_name)

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    ret_faces = []
    for (x, y, w, h) in faces:
        cx = (x+w)/2
        cy = (y+h)/2
        #ensure that face is kind of in the center
        if cx < 3*img.shape[1]/4. and cx > img.shape[1]/4.:
            patch = img[y:y+h, x:x+w,:]
            ret_faces.append(patch)

    return ret_faces


def classify_face(patch):
    img = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) / 255., (s.img_size, s.img_size))
    pred = net.predict(np.array([[img.astype(np.float32)]]))
    return pred[0]

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise Exception("Too many faces") 
    if len(rects) == 0:
        raise Exception("No face")

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def preprocess(folder):
    print "Gathering images"
    import glob
    files = glob.glob("*.png")
    files.sort()

    myfiles = {}
    for f in files:
        # detect faces, get patches
        patches = detect_faces(cv2.imread(f))
        for p in patches:
            label = classify_face(p)
            if not label in myfiles: 
                myfiles[label] = {}
            day = get_day(f)
            if not day in myfiles[label]:
                myfiles[label][day] = []

            myfiles[label][day].append(f)

    # now list them and show
    for label, daylist in myfiles.iteritems():
        days = daylist.keys()
        days.sort()
        for day in days:
            files = daylist[day]
            print "Got %d images for %s on day %s" % (len(files), s.labels[label], day)
    return myfiles

def get_day(fname):
    #complete_20160910-170959246_xc683_yc354_w116_h116.png
    return fname.split('_')[1].split('-')[0]

labeled_files = preprocess("./")

old_img = None
old_points = None

for labelidx, daylist in labeled_files.iteritems():
    out_imgs = []

    print "Computing alignments for ", s.labels[labelidx]
    days = daylist.keys()
    days.sort()
    for day in days: 
        files = daylist[day]
    
        # select the most frontal face for that day
        max_score = [-1, ""]
        for i,f in enumerate(files[:]):
            try:
                img = cv2.imread(files[i])
                points = get_landmarks(img)
                ffs = frontface_score(points)
                if ffs > max_score[0]:
                    max_score = [ffs, f]
            except Exception, e:
                print "First block, exception: ", e
                pass

        try:
            img = cv2.imread(max_score[1])
            points = get_landmarks(img)

            if old_img != None:
                transmat = transformation_from_points(old_points,points)
                out_img = warp_im(img, transmat, img.shape)

            old_img = img
            old_points = points

            out_imgs.append(out_img)
        except Exception, e:
            print "Exception: ", e
            print "With file: ", f, 
            pass


    print "Displaying"
    for i in range(10):
        for img in out_imgs:
            cv2.imshow("img", img)
            cv2.waitKey(100)
