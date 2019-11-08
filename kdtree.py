# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np
import argparse
import logging
import os
from util import calcDistance

###################### params ############################
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, default=0, help='algorithms, 0 for baseline, 1 for simple knn, 2 for softmax knn.')
parser.add_argument('--classes', type=int, default=5, help='semantic classes')
parser.add_argument('--K', type=int, default=8, help='nearest neighbors')
parser.add_argument('--batch_size', type=int, default=100, help='divided into n batchs')
parser.add_argument('--resolution_level', type=int, default=1, help='MVS resolution level')
args = parser.parse_args()

label_colours = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(0,0,0)] # BGR sequence
TYPE = args.type
CLASS = args.classes
K = args.K
batch = args.batch_size
resolution = args.resolution_level
POINT_N = 0

# path="/data1/Dataset/knn/"
# path="/data1/Dataset/pku/library/"
path="/data1/Dataset/pku/m1_resized/"

###################### log system setup ############################
logName=""
if TYPE == 0:
    logName = "{}baseline_resolution={}.txt".format(path, resolution)
elif TYPE == 1:
    logName = "{}simpleknn_resolution={}_k={}.txt".format(path, resolution, K)
else:
    logName = "{}softmaxknn_resolution={}_k={}.txt".format(path, resolution, K)
logging.basicConfig(filename=logName, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("knn start: path={}, logName={}, batch={}".format(path, logName, batch,))
print("knn start: path={}, logName={}, batch={}".format(path, logName, batch))

###################### dataset setup ############################

# fx=[5,4,3,1,8,2,9,7,0,6]
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*10
# for a in range(10):
#     prediction[a] = np.load(path + "prediction/" + str(a) + ".npy")

# fx=[35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
# 25, 24, 23, 22, 21, 20, 19 ,18, 17, 16 ,
# 5, 3, 4, 2, 0, 1, 46, 9, 47, 10,
# 48, 11, 49, 12, 50, 13, 51, 14, 52, 15,
# 53, 54, 55, 60, 61, 59, 58, 57, 56, 45,
# 8, 44, 7, 43, 6, 42, 41, 40, 39, 38, 37, 36]
# WIDTH = 4384 / resolution
# HEIGHT = 2464 / resolution
# prediction = [0]*62
# for a in range(62):
#     prediction[a] = np.load(path + "prediction/DJI010" + str(22+a) + ".jpg.npy")


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
a = 0
t = 0
while a < 288:
    filePath = path + "prediction/DJI_0" + str(285+t) + ".npy"
    if os.path.exists(filePath):
        prediction[t] = np.load(filePath)
        a+=1
        logging.info("{}loaded".format(filePath))
    t+=1

###################### functions ############################

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
    logging.info("read start")
    print("read start")
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

                line = file2.readline()
                nImage = int(line.split()[0])

                # 过滤掉只有两个视角看到的点
                if nImage < 3:
                    continue

                x.append(float(dt[1]))
                y.append(float(dt[2]))
                z.append(float(dt[3]))

                pt = np.zeros(CLASS, dtype='float32')
                for _ in range(nImage):
                    line = file2.readline()
                    imageInfo = line.split()
                    # max: 图片长度
                    if (u < int(float(imageInfo[1]))):
                        u = int(float(imageInfo[1]))
                    # max: 图片宽度
                    if (v < int(float(imageInfo[2]))):
                        v = int(float(imageInfo[2]))

                    w = int(resolution * float(imageInfo[1])) # 需要对原图降采样一半再进行预测
                    h = int(resolution * float(imageInfo[2])) # 需要对原图降采样一半再进行预测
                    # 引用越界
                    if (w >= WIDTH) or (h >= HEIGHT):
                        # logging.warning("overflow point! fx: {}".format(fx[int(imageInfo[0])]))
                        # logging.warning("overflow point! u: {}".format(int(float(imageInfo[1]))))
                        # logging.warning("overflow point! v: {}".format(int(float(imageInfo[2]))))
                        continue
                    prob = prediction[fx[int(imageInfo[0])]][h][w]
                    # logging.info("probability:{}".format(prob))

                    if softmax:
                    # 对于softmax方法，累加其概率
                        pt += prob
                    else:
                    # 对于简单方法，直接取argmax
                        pt[np.argmax(prob)] += 1

                # pt /= nImage
                p.append(pt)
                
        line = file2.readline()
    logging.info("u: {} ,v: {}".format(u,v))
    print("u: {} ,v: {}".format(u,v))
    logging.info("read finished")
    print("read finished")
    file2.close()
    return [x, y, z, p]

# 写点云到obj
def writePointCloud(x, y, z, r_new, g_new, b_new, path):
    # logging.info("x:{}".format(len(x)))
    # logging.info("y:{}".format(len(y)))
    # logging.info("z:{}".format(len(z)))
    # logging.info("r_new:{}".format(len(r_new)))
    # logging.info("g_new:{}".format(len(g_new)))
    # logging.info("b_new:{}".format(len(b_new)))
    file3 = open(path, 'a')

    for i in range(POINT_N):
        file3.write(
            'v ' + str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + ' ' + str(r_new[i]) + ' ' + str(g_new[i]) + ' ' + str(b_new[i]) + '\n')
    file3.close()


###################### main ############################

# 主函数，通过循环分批读取稠密点云，避免内存爆炸
for ii in range(batch):
    logging.info("iter: {} start".format(ii))
    print("iter: {} start".format(ii))

    # read probability and 2-3D relation
    if TYPE < 2:
        [x, y, z, p] = readTxt(ii, False)
    else:
        [x, y, z, p] = readTxt(ii, True)

    point = np.array([x, y, z]).transpose()
    POINT_N = point.shape[0]
    logging.info("points: {}".format(POINT_N))
    print("points: {}".format(POINT_N))

    tree = spatial.KDTree(point)
    logging.info("build tree finished")
    print('build tree finished')

    # knn start

    r_new = []
    g_new = []
    b_new = []


    for i in range(POINT_N):

        pnn = p[i]

        if TYPE > 0:
            d, index = tree.query(point[i], k=K)
            for j in index:
                pnn += p[j]
                distance = calcDistance(point[i], point[j])
                if (distance > 1):
                    pnn += p[j] / (distance * distance)

        label = np.argwhere(pnn == np.amax(pnn)).flatten().tolist()
        if len(label) == 1:
            r_new.append(label_colours[label[0]][2])
            g_new.append(label_colours[label[0]][1])
            b_new.append(label_colours[label[0]][0])
        else: # 存在多个最大值
            l = np.argmax(p[i])
            r_new.append(label_colours[l][2])
            g_new.append(label_colours[l][1])
            b_new.append(label_colours[l][0])

    logging.info("knn finished")
    print("knn finished")

    if TYPE == 0:
        writePointCloud(x, y, z, r_new, g_new, b_new, path + "semantic/scene_dense_baseline.obj")
    elif TYPE == 1:
        writePointCloud(x, y, z, r_new, g_new, b_new, path + "semantic/scene_dense_simple_k=" + str(K) +".obj")
    else:
        writePointCloud(x, y, z, r_new, g_new, b_new, path + "semantic/scene_dense_softmax_k=" + str(K) +".obj")
    logging.info("write finished")
    print("write finished")