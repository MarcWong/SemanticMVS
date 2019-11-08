import math

def calcDistance(point1, point2):
    a = 0.4
    # print (point1, point2)
    t = math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) + (point1[2] - point2[2]) * (point1[2] - point2[2]))
    return math.pow(math.e, t/a)