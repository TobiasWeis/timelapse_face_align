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
        print "FACE"
        # first, save patches to file
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
    
    '''
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    '''

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def preprocess(folder):
    print "Gathering images"
    import glob
    files = glob.glob("*.png")
    files.sort()

    myfiles = {}
    for f in files:
        print f
        # detect faces, get patches
        patches = detect_faces(cv2.imread(f))
        print "Got %d patches" % len(patches)
        for p in patches:
            label = classify_face(p)
            try:
                myfiles[label].append(f)
            except:
                myfiles[label] = []
                myfiles[label].append(f)
    return myfiles


labeled_files = preprocess("./")

for k,files in labeled_files.iteritems():
    out_imgs = []
    print "Computing alignments for ", s.labels[k]
    initial = True
    img1 = cv2.imread(files[0])
    for i,f in enumerate(files[:-1]):
        img2 = cv2.imread(files[i+1])

        if initial:
            out_imgs.append(img1)
            initial = False

        try:
            points1 = get_landmarks(img1)
            points2 = get_landmarks(img2)

            transmat = transformation_from_points(points1,points2)
            out_img = warp_im(img2, transmat, img1.shape)

            out_imgs.append(out_img)
        except:
            pass


    print "Displaying"
    for img in out_imgs:
        cv2.imshow("img", img)
        cv2.waitKey(100)
