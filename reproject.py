# -*- coding: utf-8 -*-
import cv2
import numpy as np
images=[]
resolution = 2
label_colours = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(0,0,0)] # BGR sequence

path="/data1/Dataset/knn/"
fx=[5,4,3,1,8,2,9,7,0,6]
WIDTH = 4000 / resolution
HEIGHT = 3000 / resolution
prediction = [0]*10
for a in range(10):
    prediction[a] = np.load(path + "prediction/DJI_02" + str(85+a) + ".npy")
for i in range(10):
    image=cv2.imread(path + "images/DJI_%04d.JPG"%(285+i))
    image=cv2.resize(image,(WIDTH, HEIGHT))
    for j in range(HEIGHT):
        for k in range(WIDTH):
            image[j][k]=[0,0,0]
    images.append(image)

# path="/data1/Dataset/pku/library/"
# fx=[35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
# 25, 24, 23, 22, 21, 20, 19 ,18, 17, 16 ,
# 5, 3, 4, 2, 0, 1, 46, 9, 47, 10,
# 48, 11, 49, 12, 50, 13, 51, 14, 52, 15,
# 53, 54, 55, 60, 61, 59, 58, 57, 56, 45,
# 8, 44, 7, 43, 6, 42, 41, 40, 39, 38, 37, 36]
# WIDTH = 4384
# HEIGHT = 2464
# prediction = [0]*62
# for a in range(62):
#     prediction[a] = np.load(path + "prediction/DJI010" + str(22+a) + ".jpg.npy")
# for i in range(62):
#     image=cv2.imread(path + "images/DJI010%02d.jpg"%(22+i))
#     image=cv2.resize(image,(WIDTH / (2 * resolution), HEIGHT / (2 * resolution)))
#     for j in range(HEIGHT / (2 * resolution)):
#         for k in range(WIDTH / (2 * resolution)):
#             image[j][k]=[0,0,0]
#     images.append(image)


# 从txt读二三维匹配点信息
def readTxt():
    file = open(path + "output.txt")
    line = file.readline()
    c=0
    while (line):
        c=c+1
        dt=line.split()

        # 3D point
        if (dt[0] == 'v'):
            indexs=[]
            xs=[]
            ys=[]
            line = file.readline()
            nImage = int(line.split()[0])

            for _ in range(nImage):
                line=file.readline()
                dt=line.split()
                index=int(dt[0])
                x=int(float(dt[1]))
                y=int(float(dt[2]))
                indexs.append(index)
                xs.append(x)
                ys.append(y)
            

            for i in range(nImage):
                # 引用越界
                w = int(resolution * float(xs[i]))
                h = int(resolution * float(ys[i]))
                if (w >= WIDTH) or (h >= HEIGHT):
                    # print('fx:', fx[int(indexs[i])])
                    # print('u:', w)
                    # print('v:', h)
                    continue

                prob = prediction[fx[indexs[i]]][h][w]

                if (w < WIDTH) and (h < HEIGHT):
                    l = np.argmax(prob)
                    images[fx[indexs[i]]][h][w]=label_colours[l]
        line=file.readline()
    print('points:', c)


print('image read finished')


# read probability from 2-3D relation Txt
readTxt()

for i in range(10):
    cv2.imwrite(path + "visualization/reproj_DJI_%04d.JPG"%(285+i),images[i])

# for i in range(62):
#     cv2.imwrite(path + "visualization/reproj_DJI010%02d.jpg"%(22+i),images[i])