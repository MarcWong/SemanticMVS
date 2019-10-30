# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np

# 从ply读点云
def readPointCloud(file):
    x = []
    y = []
    z = []
    r = []
    g = []
    b = []
    ct = 0

    line = file.readline()
    while line:
        if (ct % 100 == ii):
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

        line = file.readline()
        ct = ct + 1
    return [x, y, z, r, g, b]

# 写点云到obj
def writePointCloud(x, y, z, r_new, g_new, b_new):
    file2 = open('scene_dense2.obj', 'a')

    for i in range(point.shape[0]):
        file2.write(
            'v ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(r_new[i]) + ' ' + str(g_new[i]) + ' ' + str(b_new[i]) + '\n')
    file2.close()

# 主函数，通过循环避免内存爆炸
for ii in range(1): # 100
    file = open('scene_dense.ply')

    for i in range(13):
        line = file.readline()

    # read point cloud
    [x,y,z,r,g,b] = readPointCloud(file)
    print('iter: ', ii, 'read finished')

    point = np.array([x, y, z]).transpose()
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
        rr = r[i]
        gg = g[i]
        bb = b[i]
        flag = 0
        for j in index:
            # print(j)
            rr = rr + r[j]
            gg = gg + g[j]
            bb = bb + b[j]
    
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
            r_new.append(r[i])
            g_new.append(g[i])
            b_new.append(b[i])
            flag = 1
    print('knn finished')

    writePointCloud(x, y, z, r_new, g_new, b_new)
    print('write finished')