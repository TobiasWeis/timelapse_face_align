import math

def dist(p1,p2):
    print "Got p1: ", p1
    return math.sqrt(p1[0,0]*p1[0,0] + p2[0,0]*p2[0,0])

def frontface_score(points):
    '''
    compute how "frontal" the face is,
    as we would prefer to have good views of the face for the videos
    '''
    # use the symmetry of nose to left/right ear,
    # and nose to chin/top of head
    # #33 is nose-tip
    # #0 is right head-bondary
    # #16 is left head-boundary

    dl = float(dist(points[0],points[33]))
    dr = float(dist(points[16],points[33]))

    # want a score between 0 and 1
    return abs(dl - dr) / (dl + dr)
