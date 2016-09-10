#!/usr/bin/python

import cv2
import dlib
import numpy

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    
    '''
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    '''

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

print "Gathering images"
import glob
files = glob.glob("*.png")
files.sort()

out_imgs = []
img1 = cv2.imread(files[0])

print "Computing alignments"
initial = True
for i,f in enumerate(files[1:]):
    img2 = cv2.imread(files[i+1])

    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)

    transmat = transformation_from_points(points1,points2)
    out_img = warp_im(img2, transmat, img1.shape)

    if initial:
        out_imgs.append(img1)
        initial = False
    out_imgs.append(out_img)

print "Displaying"
for i in range(10):
    for img in out_imgs:
        cv2.imshow("img", img)
        cv2.waitKey(100)
