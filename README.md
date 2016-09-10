# timelapse_face_align
Given a bunch of images, this code warps the images in a way that detected faces are overlayed.

This algorithm uses dlib to detect the faces and keypoints,
and numpy to calculate the best warping between images.

You can get the needed face-predictor here: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
