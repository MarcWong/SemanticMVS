import numpy as np
import math

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
    return label

# 写点云到obj
def writePointCloud(x, y, z, p_new, path):
    point = np.array([x, y, z]).transpose()
    POINT_N = point.shape[0]

    file3 = open(path, 'a')
    for i in range(POINT_N):
        file3.write(
            'v ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(p_new[i][2]) + ' ' + str(p_new[i][1]) + ' ' + str(p_new[i][0]) + '\n')
    file3.close()
