# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

###################### params ############################
resolution = 2
label_colours = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(0,0,0)] # BGR sequence

# path="/data1/Dataset/knn/"
# path="/data1/Dataset/pku/library/"
path="/data1/Dataset/pku/m1_resized/"

###################### dataset setup ############################

# fx=[5,4,3,2,7,0,9,8,1,6]
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*10
# visualizations = np.zeros((10,HEIGHT,WIDTH,3), dtype='uint8')
# reprojs = np.zeros((10,HEIGHT,WIDTH), dtype='uint8')
# for i in range(10):
#     prediction[i] = np.load(path + "prediction/DJI_%04d.npy"%(285+i))

# fx=[35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
# 25, 24, 23, 22, 21, 20, 19 ,18, 17, 16 ,
# 5, 3, 4, 2, 0, 1, 46, 9, 47, 10,
# 48, 11, 49, 12, 50, 13, 51, 14, 52, 15,
# 53, 54, 55, 60, 61, 59, 58, 57, 56, 45,
# 8, 44, 7, 43, 6, 42, 41, 40, 39, 38, 37, 36]
# WIDTH = 4384 / resolution
# HEIGHT = 2464 / resolution
# prediction = [0]*62
# visualizations = np.zeros((62,HEIGHT,WIDTH,3), dtype='uint8')
# reprojs = np.zeros((62,HEIGHT,WIDTH), dtype='uint8')
# for i in range(62):
#     prediction[i] = np.load(path + "prediction/DJI010%02d.jpg.npy"%(22+i))

fx=[193, 192, 191, 190, 189, 188, 187, 186 ,185, 184, 183, 182, 181, 180, 179,
178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163,
162, 161, 160, 159, 158, 157, 156, 153, 152, 151, 150, 149, 148, 147, 146, 145,
144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129,
128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113,
112, 111, 110, 109, 108, 107, 106, 105, 104, 35, 34, 33, 32, 31, 30, 29, 28, 27,
26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 5, 4, 3, 2, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13,
238, 43, 239, 44, 240, 45, 241, 46, 242, 47, 243, 48, 244, 49, 245, 50, 246, 51,
247, 52, 248, 53, 249, 54, 250, 55, 251, 56, 252, 57, 253, 58, 254, 59, 255, 60,
256, 61, 257, 62, 258, 63, 259, 64, 260, 65, 261, 66, 262, 67, 263, 94, 264, 95,
265, 96, 266, 97, 267, 98, 268, 99, 269, 100, 270, 101, 271, 102, 272, 103, 273,
274, 275, 276, 277, 278, 279, 300, 301, 302, 303, 304, 305, 306, 307, 308,
326, 331, 332, 330, 329, 328, 327, 299, 298, 297, 296, 295, 294, 293, 292, 291,
290, 289, 288, 287, 286, 285, 284, 283, 282, 281, 280, 237, 39, 236, 38, 235, 37, 234, 36,
233, 35, 232, 34, 231, 33, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219,
218, 217, 216, 215, 214, 213, 212, 211, 210, 15, 209, 14, 208, 207, 206, 205, 204,
203, 202, 201, 200, 199, 198, 197, 196, 195, 194]
WIDTH = 4000 / resolution
HEIGHT = 3000 / resolution
prediction = [0]*333
visualizations = np.zeros((333,HEIGHT,WIDTH,3), dtype='uint8')
reprojs = np.zeros((333,HEIGHT,WIDTH), dtype='uint8')
a = 0
t = 0
while a < 288:
    filePath = path + "prediction/DJI_0" + str(285+t) + ".npy"
    if os.path.exists(filePath):
        prediction[t] = np.load(filePath)
        a+=1
    t+=1

###################### functions ############################

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
                    continue

                prob = prediction[fx[indexs[i]]][h][w]
                if (w < WIDTH) and (h < HEIGHT):
                    l = np.argmax(prob)
                    visualizations[fx[indexs[i]]][h][w]=label_colours[l]
                    reprojs[fx[indexs[i]]][h][w]=l
        line=file.readline()
    print('points:', c)

print('image read finished')


# read probability from 2-3D relation Txt
readTxt()

# for i in range(10):
#     cv2.imwrite(path + "reproj/visual_DJI_%04d.JPG"%(285+i),visualizations[i])
#     cv2.imwrite(path + "reproj/reproj_DJI_%04d.png"%(285+i),reprojs[i])

# for i in range(62):
    # cv2.imwrite(path + "reproj/visual_DJI010%02d.JPG"%(22+i),visualizations[i])
    # cv2.imwrite(path + "reproj/reproj_DJI010%02d.png"%(22+i),reprojs[i])

a = 0
t = 0
while a < 288:
    filePath = path + "prediction/DJI_0" + str(285+t) + ".npy"
    if os.path.exists(filePath):
        cv2.imwrite(path + "reproj/visual_DJI_%04d.JPG"%(285+t),visualizations[t])
        cv2.imwrite(path + "reproj/reproj_DJI_%04d.png"%(285+t),reprojs[t])
        a+=1
    t+=1
