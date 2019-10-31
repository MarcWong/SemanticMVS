# -*- coding: utf-8 -*-
import cv2
import numpy as np
images=[]

fx=[5,4,3,1,8,2,9,7,0,6]

prediction = [0]*10
for a in range(10):
    prediction[a] = np.load("/data1/Dataset/knn/prediction/"+str(a)+".npy")

# 从txt读二三维匹配点信息
def readTxt():
    file=open("/data1/Dataset/knn/output.txt")
    line=file.readline()
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
                [prob0, prob1, prob2] = prediction[fx[int(indexs[i])]][2 * int(float(ys[i]))][2 * int(float(xs[i]))]
                if (xs[i]<2000 and ys[i]<1500):
                    if (prob0 > prob1) and (prob0 > prob2):
                        images[fx[indexs[i]]][ys[i]][xs[i]][0]=0
                        images[fx[indexs[i]]][ys[i]][xs[i]][1]=255
                        images[fx[indexs[i]]][ys[i]][xs[i]][2]=0
                    if (prob1 > prob0) and (prob1 > prob2):
                        images[fx[indexs[i]]][ys[i]][xs[i]][0]=0
                        images[fx[indexs[i]]][ys[i]][xs[i]][1]=0
                        images[fx[indexs[i]]][ys[i]][xs[i]][2]=255
                    if (prob2 > prob1) and (prob2 > prob0):
                        images[fx[indexs[i]]][ys[i]][xs[i]][0]=255
                        images[fx[indexs[i]]][ys[i]][xs[i]][1]=0
                        images[fx[indexs[i]]][ys[i]][xs[i]][2]=0
        line=file.readline()
    print('points:', c)


for i in range(10):
    image=cv2.imread("/data1/Dataset/knn/images/DJI_%04d.JPG"%(285+i))
    image=cv2.resize(image,(2000,1500))
    for j in range(1500):
        for k in range(2000):
            image[j][k]=[0,0,0]
    images.append(image)

print('image read finished')

# read probability from 2-3D relation Txt
readTxt()

for i in range(10):
    cv2.imwrite("/data1/Dataset/knn/visualization/reproj_DJI_%04d.JPG"%(285+i),images[i])
