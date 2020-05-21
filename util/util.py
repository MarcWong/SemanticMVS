
import matplotlib.pyplot as plt
import numpy as np
import math
import re
label_colours = [[35,142,107],[70,70,70],[128,64,128],[0,0,142],[0,0,0]] # RGB sequence, # 0=vegetarian, 1=building, 2=road 3=vehicle, 4=other

def calcDistance(point1, point2):
    a = 0.4
    # print (point1, point2)
    t = math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) + (point1[2] - point2[2]) * (point1[2] - point2[2]))
    return math.pow(math.e, t/a)

def rgb2label(bgr_array):
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

def readfile(filename):
    fx = []
    with open(filename) as f:
        for line in f.readlines():
            line_str = line.strip()
            line_list = line_str.split(",")
            for i in line_list:
                if i:
                    fx.append(int(i))

    return fx


def generateTXT(flogPath, foutPath):
    
    flog=open(flogPath)
    fout=open(foutPath, "w")

    line = flog.readline()
    while (line):
        if 'DJI_' in line:
            a = re.findall(r'DJI_[^c]*.JPG', line)[0][4:8]
            # print(a)
            fout.write(a)
            fout.write(',')
        line = flog.readline()


def draw_figure():
    fig = plt.figure(figsize=(16,12))
    x = (4, 5, 6, 8, 10, 15, 20)
    y1 = [87.72, 87.73, 87.74, 87.89, 87.91, 88.23, 87.32]
    y2 = [87.64, 87.62, 87.62, 87.84, 87.86, 88.14, 87.37]
    plt.plot(x,y1,label='Prob')
    plt.plot(x,y2,label='Argmax')
    plt.xticks(x,fontsize=36)#设置x的刻度
    plt.yticks(fontsize=36)#设置x的刻度
    # plt.xticks(x[::2])
    plt.ylabel('Accuracy(%)',fontsize=40)
    plt.xlabel('k',fontsize=40)

    plt.legend(loc=2, ncol=3, fontsize=34)

    plt.savefig('ablation_k.png', format='png')

