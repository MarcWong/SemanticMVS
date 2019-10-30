# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np

batch = 50

# 从ply读点云
def readPointCloud(ii):
    file1 = open('model_dense.ply')

    for _ in range(13):
        line = file1.readline()
    x = []
    y = []
    z = []
    r = []
    g = []
    b = []
    ct = 0

    line = file1.readline()
    while line:
        if (ct % batch == ii):
            dt = line.split()

            # run sparse
            # x.append(float(dt[1]))
            # y.append(float(dt[2]))
            # z.append(float(dt[3]))
            # r.append(int(dt[4]))
            # g.append(int(dt[5]))
            # b.append(int(dt[6]))

            # run dense
            x.append(float(dt[0]))
            y.append(float(dt[1]))
            z.append(float(dt[2]))
            r.append(int(dt[3]))
            g.append(int(dt[4]))
            b.append(int(dt[5]))

        line = file1.readline()
        ct += 1
    file1.close()
    return [x, y, z, r, g, b]

# 从txt读二三维匹配点信息
def readTxt(ii):
    file2 = open('/data1/Dataset/knn/output.txt')
    x = []
    y = []
    z = []
    p = []
    ct = 0

    u=0
    v=0
    line = file2.readline()
    while line:
        dt = line.split()
        # 3D point
        if (dt[0] == 'v'):
            ct += 1
            # add this point to KDtree
            if (ct % batch == ii):
                x.append(float(dt[1]))
                y.append(float(dt[2]))
                z.append(float(dt[3]))

                line = file2.readline()
                nImage = int(line.split()[0])

                p0 = 0
                p1 = 0
                p2 = 0
                for _ in range(nImage):
                    line = file2.readline()
                    imageInfo = line.split()
                    # max: 2000
                    if (u < int(float(imageInfo[1]))):
                        u = int(float(imageInfo[1]))
                    # max: 1500
                    if (v < int(float(imageInfo[2]))):
                        v = int(float(imageInfo[2]))
                    # print(prediction[int(imageInfo[0])].shape)
                    # print('u:', int(float(imageInfo[1])))
                    # print('v:', int(float(imageInfo[2])))
                    [prob0, prob1, prob2] = prediction[int(imageInfo[0])][2 * int(float(imageInfo[2]))][2 * int(float(imageInfo[1]))]
                    # print('probability:', [prob0, prob1, prob2])
                    p0 += prob0
                    p1 += prob1
                    p2 += prob2
                p0 /= nImage
                p1 /= nImage
                p2 /= nImage
                p.append([p0, p1, p2])

        line = file2.readline()
    print('u:', u)
    print('v:', v)
    file2.close()
    return [x, y, z, p]

# 写点云到obj
def writePointCloud(x, y, z, r_new, g_new, b_new):
    file3 = open('scene_dense2.obj', 'a')

    for i in range(point.shape[0]):
        file3.write(
            'v ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(r_new[i]) + ' ' + str(g_new[i]) + ' ' + str(b_new[i]) + '\n')
    file3.close()


prediction = [0]*10
for a in range(10):
    prediction[a] = np.load("/data1/Dataset/knn/prediction/"+str(a)+".npy")

# 主函数，通过循环分批读取稠密点云，避免内存爆炸
for ii in range(batch):
    print("iter: ", ii, "start")

    # read point cloud
    # [x,y,z,r,g,b] = readPointCloud(ii)

    # read probability from 2-3D relation Txt
    [x2,y2,z2, p] = readTxt(ii)

    point = np.array([x2, y2, z2]).transpose()
    print(point.shape[0])

    tree = spatial.KDTree(point)
    print('build tree finished')

    # knn start

    r_new = []
    g_new = []
    b_new = []

    k = int(point.shape[0] / 1000)

    for i in range(point.shape[0]):
        d, index = tree.query(point[i], k=10)
        rr = p[i][2]
        gg = p[i][1]
        bb = p[i][0]
        flag = 0
        for j in index:
            # print(j)
            rr += p[j][2]
            gg += p[j][1]
            bb += p[j][0]

        # print('rr:', rr)
        # print('gg:', gg)
        # print('bb:', bb)
    
        if (rr > gg) and (rr > bb):
            r_new.append(255)
            g_new.append(0)
            b_new.append(0)
            flag = 1
        if (gg > rr) and (gg > bb):
            r_new.append(0)
            g_new.append(255)
            b_new.append(0)
            flag = 1
        if (bb > gg) and (bb > rr):
            r_new.append(0)
            g_new.append(0)
            b_new.append(255)
            flag = 1
        if (flag == 0):
            if (p[i][2] > p[i][1] and p[i][2] > p[i][0]):
                r_new.append(255)
                g_new.append(0)
                b_new.append(0)
            if (p[i][1] > p[i][2] and p[i][1] > p[i][0]):
                r_new.append(0)
                g_new.append(255)
                b_new.append(0)
            if (p[i][0] > p[i][2] and p[i][0] > p[i][1]):
                r_new.append(0)
                g_new.append(0)
                b_new.append(255)
            flag = 1
    print('knn finished')

    writePointCloud(x2, y2, z2, r_new, g_new, b_new)
    print('write finished')