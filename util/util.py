import numpy as np
import math
label_colours = [(107,142,35),(70,70,70),(128,64,128),(142,0,0),(0,0,0)] # RGB sequence, # 0=vegetarian, 1=building, 2=road 3=vehicle, 4=other

def calcDistance(point1, point2):
    a = 0.4
    # print (point1, point2)
    t = math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) + (point1[2] - point2[2]) * (point1[2] - point2[2]))
    return math.pow(math.e, t/a)

def bgr2label(bgr_array):
    label = -1
    for i in range(len(label_colours)):
        if label_colours[i] == bgr_array:
            label = i
            break
    if label == -1:
        print(bgr_array)
    return label

# 写点云到obj
def writePointCloud(points_new, p_new, path):
    file3 = open(path, 'a')
    for i in range(len(points_new)):
        file3.write('v ' + str(points_new[i][0]) + ' ' + str(points_new[i][1]) + ' ' + str(points_new[i][2]) + ' ' + str(p_new[i][2]) + ' ' + str(p_new[i][1]) + ' ' + str(p_new[i][0]) + '\n')
    file3.close()
