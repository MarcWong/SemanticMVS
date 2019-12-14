import numpy as np
import math
label_colours = [(35,142,107),(70,70,70),(128,64,128),(0,0,142),(0,0,0)] # BGR sequence, # 0=vegetarian, 1=building, 2=road 3=vehicle, 4=other

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
    print(len(x),len(y),len(p_new[0]),len(p_new[1]),len(p_new[2]))
    file3 = open(path, 'a')
    for i in range(POINT_N):
        file3.write(
            'v ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(p_new[2][i]) + ' ' + str(p_new[1][i]) + ' ' + str(p_new[0][i]) + '\n')
    file3.close()
