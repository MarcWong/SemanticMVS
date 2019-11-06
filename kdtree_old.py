# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np

# 0: baseline
# 1: simple knn
# 2: argmax knn
TYPE = 0

# nearest neighbor
K = 8

batch = 50
POINT_N = 0
resolution = 1

path="/data1/Dataset/knn/"
fx=[5,4,3,1,8,2,9,7,0,6]
WIDTH = 4000
HEIGHT = 3000
prediction = [0]*10
for a in range(10):
    prediction[a] = np.load(path + "prediction/" + str(a) + ".npy")

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

# 从ply读点云，暂时废弃
def readPointCloud(ii):
    file1 = open('sparse_dense.ply')

    for _ in range(13):
        line = file1.readline()
    x = []
    y = []
    z = []
    p = []

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
            
            print('x:', dt[0])
            x.append(float(dt[0]))
            y.append(float(dt[1]))
            z.append(float(dt[2]))
            p.append([int(dt[3]) / 255., int(dt[4]) / 255., int(dt[5]) / 255.])

        line = file1.readline()
        ct += 1
    file1.close()
    return [x, y, z, p]

# 从txt读二三维匹配点信息，投票
def readTxt(ii, softmax):
    file2 = open(path + "output.txt")
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

                p0 = 0.0
                p1 = 0.0
                p2 = 0.0
                for _ in range(nImage):
                    line = file2.readline()
                    imageInfo = line.split()
                    # max: 图片长度的1/2
                    if (u < int(float(imageInfo[1]))):
                        u = int(float(imageInfo[1]))
                    # max: 图片宽度的1500
                    if (v < int(float(imageInfo[2]))):
                        v = int(float(imageInfo[2]))

                    w = 2 * int(resolution * float(imageInfo[1]))
                    h = 2 * int(resolution * float(imageInfo[2]))
                    # 引用越界
                    if (w >= WIDTH) or (h >= HEIGHT):                       
                        # print('fx:', fx[int(imageInfo[0])])
                        # print('u:', int(float(imageInfo[1])))
                        # print('v:', int(float(imageInfo[2])))
                        continue
                    [prob0, prob1, prob2] = prediction[fx[int(imageInfo[0])]][h][w]
                    # print('probability:', [prob0, prob1, prob2])

                    if softmax:
                    # 对于softmax方法，累加其概率
                        p0 += prob0
                        p1 += prob1
                        p2 += prob2
                    else:
                    # 对于简单方法，直接取argmax
                        if (prob0 >= prob1 and prob0 >= prob2):
                            p0 += 1
                        elif (prob1 >= prob0 and prob1 >= prob2):
                            p1 += 1
                        elif (prob2 >= prob0 and prob2 >= prob1):
                            p2 += 1

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
def writePointCloud(x, y, z, r_new, g_new, b_new, path):
    # print('x:', len(x))
    # print('y:', len(y))
    # print('z:', len(z))
    # print('r_new:', len(r_new))
    # print('g_new:', len(g_new))
    # print('b_new:', len(b_new))
    file3 = open(path, 'a')

    for i in range(POINT_N):
        file3.write(
            'v ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(r_new[i]) + ' ' + str(g_new[i]) + ' ' + str(b_new[i]) + '\n')
    file3.close()


# 主函数，通过循环分批读取稠密点云，避免内存爆炸
for ii in range(batch):
    print("iter: ", ii, "start")

    # read probability and 2-3D relation
    if TYPE < 2:
        [x, y, z, p] = readTxt(ii, False)
    else:
        [x, y, z, p] = readTxt(ii, True)

    point = np.array([x, y, z]).transpose()
    POINT_N = point.shape[0]
    print(POINT_N)

    tree = spatial.KDTree(point)
    print('build tree finished')

    # knn start

    r_new = []
    g_new = []
    b_new = []


    for i in range(POINT_N):
        d, index = tree.query(point[i], k=K)

        gg = p[i][0]
        rr = p[i][1]
        bb = p[i][2]
        
        flag = 0
        if TYPE > 0:
            for j in index:
                rr += p[j][1]
                gg += p[j][0]
                bb += p[j][2]

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
            if (p[i][0] >= p[i][2] and p[i][0] >= p[i][1]):
                r_new.append(0)
                g_new.append(255)
                b_new.append(0)
            elif (p[i][1] >= p[i][2] and p[i][1] >= p[i][0]):
                r_new.append(255)
                g_new.append(0)
                b_new.append(0)
            elif (p[i][2] >= p[i][1] and p[i][2] >= p[i][0]):
                r_new.append(0)
                g_new.append(0)
                b_new.append(255)
            else:
                r_new.append(255)
                g_new.append(255)
                b_new.append(255)
            flag = 1
    print('knn finished')

    if TYPE == 0:
        writePointCloud(x, y, z, r_new, g_new, b_new, path + "semantic/scene_dense_baseline.obj")
    elif TYPE == 1:
        writePointCloud(x, y, z, r_new, g_new, b_new, path + "semantic/scene_dense_simple_k=" + str(K) +".obj")
    else:
        writePointCloud(x, y, z, r_new, g_new, b_new, path + "semantic/scene_dense_softmax_k=" + str(K) +".obj")
    print('write finished')