# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np
import argparse
import logging
import os
import operator
from util.util import calcDistance, writePointCloud, rgb2label, readfile, generateTXT

###################### params ############################
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, default=0, help='algorithms, 0 for baseline, 1 for simple knn, 2 for softmax knn, 3 for energy min.')
parser.add_argument('--classes', type=int, default=5, help='semantic classes')
parser.add_argument('--K', type=int, default=8, help='nearest neighbors')
parser.add_argument('--batch_size', type=int, default=100, help='divided into n batchs')
parser.add_argument('--obj_path', type=str, default="semantic/result/prob/rate/scene_dense_softmax_k=15batch_size=1.obj", help='semantic point cloud path')
parser.add_argument('--resolution_level', type=int, default=1, help='MVS resolution level')
args = parser.parse_args()

obj_path = args.obj_path
label_colours = [(107,142,35),(70,70,70),(128,64,128),(0,0,142),(0,0,0)] # BGR sequence, # 0=vegetarian, 1=building, 2=road 3=vehicle, 4=other
TYPE = args.type
CLASS = args.classes
K = args.K
batch = args.batch_size
resolution = args.resolution_level
POINT_N = 0

###################### dataset setup ############################

# PKU library
# path="/data1/Dataset/pku/library/"
# fx=readfile("/data1/wy/semantic-point-cloud/data/PKU-Library.txt")
# WIDTH = 4384 / resolution
# HEIGHT = 2464 / resolution
# prediction = [0]*62
# for a in range(62):
#     prediction[a] = np.load(path + "prediction/DJI010" + str(22+a) + ".jpg.npy")

# PKU M1
# path="/data1/Dataset/pku/m1_semantic/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/PKU-M1.txt")
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*333
# a = 0
# t = 0
# while a < 288:
#    filePath = path + "prediction/DJI_%04d.npy"%(t+285)
#    if os.path.exists(filePath):
#        prediction[t] = np.load(filePath)
#        a+=1
#        logging.info("{}loaded".format(filePath))
#    t+=1


# PKU N1
# path="/data1/Dataset/pku/n1/"
# fx=readfile("/data1/wy/semantic-point-cloud/data/PKU-N1.txt")
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*999
# a = 0
# t = 0
# while a < 350:
#    filePath = path + "prediction/DJI_%04d.npy"%(t+1)

#    if os.path.exists(filePath):
#        prediction[t] = np.load(filePath)
#        a+=1
#        logging.info("{}loaded".format(filePath))
#    t+=1


# PKU E44
# path="/data1/Dataset/pku/e44/"
# logPath="/data1/Dataset/pku/e44/dense/dense/DensifyPointCloud-2005200355358B5E96.log"
# outPath="/data1/wy/semantic-point-cloud/data/PKU-E44.txt"

# generateTXT(logPath, outPath)
# fx=readfile(outPath)
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*999
# a = 0
# t = 0
# while a < 337:
#    filePath = path + "prediction/DJI_%04d.npy"%(t)

#    if os.path.exists(filePath):
#        prediction[t] = np.load(filePath)
#        a+=1
#        logging.info("{}loaded".format(filePath))
#    t+=1

# Contest Tianjin
# path="/data1/Dataset/Semantic/game/Tianjin_semantic/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/Tianjin.txt")
# WIDTH = 2736 / resolution
# HEIGHT = 1824 / resolution
# prediction = [0]*1000
# a = 0
# t = 0
# while a < 484:
#     filePath = path + "prediction/DJI_%04d.npy"%(t+1)
#     if os.path.exists(filePath):
#         prediction[t] = np.load(filePath)
#         a+=1
#         logging.info("{}loaded".format(filePath))
#     t+=1

# HU-Hall
# path="/data1/Dataset/heda/dalitang/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/HU-Hall.txt")
#WIDTH = 4000 / resolution
#HEIGHT = 3000 / resolution
#prediction = [0]*999
#a = 0
#t = 0
#while a < 195:
#   filePath = path + "prediction/DJI_%04d.npy"%(t)
#   if os.path.exists(filePath):
#       prediction[t] = np.load(filePath)
#       a+=1
#       logging.info("{}loaded".format(filePath))
#   t+=1

# HU-Anyuanmen
# path="/data1/Dataset/heda/anyuanmen/"
# logPath="/data1/Dataset/heda/anyuanmen/dense/dense/DensifyPointCloud-2005210032278B81C0.log"
# outPath="/data1/wy/semantic-point-cloud/data/HU-Anyuanmen.txt"

# generateTXT(logPath, outPath)
# fx=readfile(outPath)
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*999
# a = 0
# t = 0
# while a < 148:
#   filePath = path + "prediction/DJI_%04d.npy"%(t)
#   if os.path.exists(filePath):
#       prediction[t] = np.load(filePath)
#       a+=1
#       logging.info("{}loaded".format(filePath))
#   t+=1

# HU-Jieyindian
# path="/data1/Dataset/heda/jieyindian/"
# logPath="/data1/Dataset/heda/jieyindian/dense/dense/DensifyPointCloud-2005210032278B81C0.log"
# outPath="/data1/wy/semantic-point-cloud/data/HU-Jieyindian.txt"

# generateTXT(logPath, outPath)
# fx=readfile(outPath)
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*999
# a = 0
# t = 0
# while a < 86:
#   filePath = path + "prediction/DJI_%04d.npy"%(t)
#   if os.path.exists(filePath):
#       prediction[t] = np.load(filePath)
#       a+=1
#       logging.info("{}loaded".format(filePath))
#   t+=1

# HU-Longting
# path="/data1/Dataset/heda/longting/"
# logPath="/data1/Dataset/heda/longting/dense/dense/DensifyPointCloud-2005210251188B8AE7.log"
# outPath="/data1/wy/semantic-point-cloud/data/HU-Longting.txt"

# generateTXT(logPath, outPath)
# fx=readfile(outPath)
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*999
# a = 0
# t = 0
# while a < 135:
#   filePath = path + "prediction/DJI_%04d.npy"%(t)
#   if os.path.exists(filePath):
#       prediction[t] = np.load(filePath)
#       a+=1
#       logging.info("{}loaded".format(filePath))
#   t+=1

# HU-Library
# path="/data1/Dataset/heda/library/"
# logPath="/data1/Dataset/heda/library/dense/dense/DensifyPointCloud-2005210308108BAF5B.log"
# outPath="/data1/wy/semantic-point-cloud/data/HU-Library.txt"

# generateTXT(logPath, outPath)
# fx=readfile(outPath)
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*1000
# a = 0
# t = 0
# while a < 225:
#   filePath = path + "prediction/DJI_%04d.npy"%(t)
#   if os.path.exists(filePath):
#       prediction[t] = np.load(filePath)
#       a+=1
#       logging.info("{}loaded".format(filePath))
#   t+=1


# HU-Shizhai
path="/data1/Dataset/heda/shizhai/"
logPath="/data1/Dataset/heda/shizhai/dense/dense/DensifyPointCloud-2005211115438BB885.log"
outPath="/data1/wy/semantic-point-cloud/data/HU-Shizhai.txt"

generateTXT(logPath, outPath)
fx=readfile(outPath)
WIDTH = 4000 / resolution
HEIGHT = 3000 / resolution
prediction = [0]*1000
a = 0
t = 0
while a < 86:
  filePath = path + "prediction/DJI_%04d.npy"%(t)
  if os.path.exists(filePath):
      prediction[t] = np.load(filePath)
      a+=1
      logging.info("{}loaded".format(filePath))
  t+=1

# nanshao_under_construction
# path="/data1/Dataset/Henan/nanshao_under_construction/"
# logPath="/data1/Dataset/Henan/nanshao_under_construction/dense/dense/DensifyPointCloud-2005202136088B6785.log"
# outPath="/data1/wy/semantic-point-cloud/data/nanshao_under_construction.txt"

# generateTXT(logPath, outPath)
# fx=readfile(outPath)
# WIDTH = 4056 / resolution
# HEIGHT = 3040 / resolution
# prediction = [0]*999
# a = 0
# t = 0
# while a < 115:
#    filePath = path + "prediction/DJI_%04d.npy"%(t)

#    if os.path.exists(filePath):
#        prediction[t] = np.load(filePath)
#        a+=1
#        logging.info("{}loaded".format(filePath))
#    t+=1


# TNT-Family
# path="/data1/Dataset/Benchmark/tanksandtemples/test/Family/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-Family.txt")
#WIDTH = 1920
#HEIGHT = 1080

# TNT-Francis
#path="/data1/Dataset/Benchmark/tanksandtemples/test/Francis/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-Francis.txt")
#WIDTH = 1920
#HEIGHT = 1080

# TNT-Horse
#path="/data1/Dataset/Benchmark/tanksandtemples/test/Horse/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-Horse.txt")
#WIDTH = 1920
#HEIGHT = 1080

# TNT-Lighthouse
# path="/data1/Dataset/Benchmark/tanksandtemples/test/Lighthouse/"
# fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-Lighthouse.txt")
# WIDTH = 2048
# HEIGHT = 1080

# TNT-M60
#path="/data1/Dataset/Benchmark/tanksandtemples/test/M60/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-M60.txt")
#WIDTH = 2048
#HEIGHT = 1080

# TNT-Panther
# path="/data1/Dataset/Benchmark/tanksandtemples/test/Panther/"
# fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-Panther.txt")
# WIDTH = 2048
# HEIGHT = 1080

# TNT-Train
#path="/data1/Dataset/Benchmark/tanksandtemples/test/Train/"
#fx=readfile("/data1/wy/semantic-point-cloud/data/TNT-Train.txt")
#WIDTH = 1920
#HEIGHT = 1080

# prediction = [0]* (len(fx)+1)
# a = 1

# print(len(fx))
# while a < len(fx):
#     filePath = path + "prediction/%05d.npy"%(a)
#     prediction[a] = np.load(filePath)
#     a+=1
#     logging.info("{}loaded".format(filePath))


###################### log system setup ############################
logName=""
if TYPE == 0:
    logName = "{}baseline_resolution={}_downsample={}.txt".format(path, resolution, batch)
elif TYPE == 1:
    logName = "{}argmax_knn_resolution={}_k={}_downsample={}.txt".format(path, resolution, K, batch)
elif TYPE == 2:
    logName = "{}prob_knn_resolution={}_k={}_downsample={}.txt".format(path, resolution, K, batch)
else:
    logName = "{}energy_resolution={}_downsample={}.txt".format(path, resolution, batch)

logging.basicConfig(filename=logName, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Algorithm begin: path={}, logName={}".format(path, logName,))
print("Algorithm begin: path={}, logName={}".format(path, logName))

###################### functions ############################

# 从ply读点云
def readPointCloud(ii):

    filePath = path + obj_path
    if os.path.exists(filePath):
        file1 = open(filePath)
    else:
        print('no such file: {}'.format(filePath))
        return

    # for _ in range(13):
    #     line = file1.readline()

    line = file1.readline()
    x = []
    y = []
    z = []
    p = []

    ct = 0
    while line:
        if (ct % batch == ii):
            dt = line.split()

            # run dense
            x.append(float(dt[1]))
            y.append(float(dt[2]))
            z.append(float(dt[3]))
            p.append([int(dt[4]), int(dt[5]), int(dt[6])])

        line = file1.readline()
        ct += 1

    file1.close()
    return [x, y, z, p]

# 从txt读二三维匹配点信息
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
                # if nImage < 3:
                #     continue


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
                    if (w >= WIDTH) or (h >= HEIGHT) or type(prediction[fx[int(imageInfo[0])]]) == int:
                        # logging.warning("overflow point! fx: {}".format(fx[int(imageInfo[0])]))
                        # logging.warning("overflow point! u: {}".format(int(float(imageInfo[1]))))
                        # logging.warning("overflow point! v: {}".format(int(float(imageInfo[2]))))
                        continue
                    prob = prediction[fx[int(imageInfo[0])]][h][w]
                    # logging.info("probability:{}".format(prob))

                    if softmax:
                    # for Prob, accumulate the prob array
                        pt += prob
                    else:
                    # for Simple, add 1 to argmax index
                        pt[np.argmax(prob)] += 1


                # randomly discard background point
                if TYPE == 0 or np.argmax(pt) != 0 or np.random.rand() > 0.95:
                    x.append(float(dt[1]))
                    y.append(float(dt[2]))
                    z.append(float(dt[3]))
                    # pt /= nImage
                    p.append(pt)

        line = file2.readline()
    logging.info("u: {} ,v: {}".format(u,v))
    print("u: {} ,v: {}".format(u,v))
    logging.info("read finished")
    print("read finished")
    file2.close()
    return [x, y, z, p]

def knn_fusion(x, y, z, p):
    point = np.array([x, y, z]).transpose()
    POINT_N = point.shape[0]
    logging.info("points: {}".format(POINT_N))
    print("points: {}".format(POINT_N))

    tree = spatial.KDTree(point)
    logging.info("build tree finished")
    print('build tree finished')

    r_new = [0] * len(x)
    g_new = [0] * len(x)
    b_new = [0] * len(x)
    able = [1] * len(x)


    for i in range(POINT_N):

        root = p[i]

        if TYPE > 0:
            if K > 1:
                _, index = tree.query(point[i], k=K)
                for j in index:
                    # use the e distance
                    distance = calcDistance(point[i], point[j])
                    if (distance > 1):
                        for m in range(len(root)):
                            root[m] += p[j][m] / (distance * distance)

        label = np.argwhere(root == np.amax(root)).flatten().tolist()
        if len(label) == 1:
            r_new[i] = label_colours[label[0]][2]
            g_new[i] = label_colours[label[0]][1]
            b_new[i] = label_colours[label[0]][0]
        else: # 存在多个最大值
            l = np.argmax(p[i])
            r_new[i] = label_colours[l][2]
            g_new[i] = label_colours[l][1]
            b_new[i] = label_colours[l][0]

    logging.info("refine finished")
    print("refine finished")

    result_xyz = []
    result_rgb = []
    for m in range(len(able)):
        if able[m]:
            result_xyz.append([x[m] , y[m], z[m]])
            result_rgb.append([r_new[m], g_new[m], b_new[m]])
    return result_xyz, result_rgb

# read from output.txt
# def energy_fusion(x, y, z, p):
#     point = np.array([x, y, z]).transpose()
#     POINT_N = point.shape[0]
#     logging.info("points: {}".format(POINT_N))
#     print("points: {}".format(POINT_N))

#     tree = spatial.KDTree(point)
#     logging.info("build tree finished")
#     print('build tree finished')

#     # dsum = 0
#     # k = int(POINT_N / 1000)
#     # for i in range(POINT_N):
#     #    d, index = tree.query(point[i], k=K)
#     #    for dk in d:
#     #         dsum+=dk
#     # dsum=dsum/POINT_N/10
#     # print(dsum)
#     dsum = 0.05

#     visit = [0] * len(x)
#     able = [1] * len(x)

#     r_new = [0] * len(x)
#     g_new = [0] * len(x)
#     b_new = [0] * len(x)


#     for i in range(POINT_N):

#         if (visit[i] == 0):
#             visit[i] = 1
#             root = p[i]

#             queue = []
#             all = []
#             queue.append(point[i])
#             all.append(i)
#             while (len(queue) > 0):
#                 temp = []
#                 for j in range(len(queue)):
#                     d, index = tree.query(queue[j], k=K)
#                     dk = np.array(d)
#                     ct = 0
#                     for k in index:
#                         if (dk[ct] < dsum) and (ct > 0):
#                             for m in range(len(root)):
#                                 root[m] += p[k][m]

#                             if (visit[k] == 0):
#                                 if np.argmax(p[k]) == np.argmax(p[i]):
#                                     temp.append(point[k])
#                                     all.append(k)
#                                     visit[k] = 1
#                         ct += 1
#                 queue = temp

#             for j in range(len(all)):
#                 # if len(all) < 3:
#                 #     able[all[j]] = 0
#                 label = np.argwhere(root == np.amax(root)).flatten().tolist()
#                 if len(label) == 1:
#                     r_new[all[j]] = label_colours[label[0]][2]
#                     g_new[all[j]] = label_colours[label[0]][1]
#                     b_new[all[j]] = label_colours[label[0]][0]
#                 else: # 存在多个最大值
#                     l = np.argmax(p[all[j]])
#                     r_new[all[j]] = label_colours[l][2]
#                     g_new[all[j]] = label_colours[l][1]
#                     b_new[all[j]] = label_colours[l][0]

#     logging.info("refine finished")
#     print('refine finished')

#     result_xyz = []
#     result_rgb = []
#     print('able:', len(able))
#     for m in range(len(able)):
#         if able[m]:
#             result_xyz.append([x[m] , y[m], z[m]])
#             result_rgb.append([r_new[m], g_new[m], b_new[m]])
#     return result_xyz, result_rgb


# read from obj
def energy_fusion(x, y, z, p):
    point = np.array([x, y, z]).transpose()
    POINT_N = point.shape[0]
    logging.info("points: {}".format(POINT_N))
    print("points: {}".format(POINT_N))

    tree = spatial.KDTree(point)
    logging.info("build tree finished")
    print('build tree finished')

    dsum = 0
    k = int(POINT_N / 100)
    for i in range(POINT_N):
        d, index = tree.query(point[i], k=K)
        for dk in d:
            dsum+=dk
    dsum=dsum/POINT_N/10
    print(dsum)
    # dsum = 0.05

    visit = [0] * len(x)
    able = [1] * len(x)

    r_new = [0] * len(x)
    g_new = [0] * len(x)
    b_new = [0] * len(x)


    for i in range(POINT_N):

        if (visit[i] == 0):
            visit[i] = 1
            root = [0] * 5

            root_label = rgb2label(p[i])
            root[root_label] += 1

            queue = []
            all = []
            queue.append(point[i])
            all.append(i)
            while (len(queue) > 0):
                temp = []
                for j in range(len(queue)):
                    d, index = tree.query(queue[j], k=K)
                    dk = np.array(d)
                    ct = 0
                    for k in index:
                        if (dk[ct] < dsum) and (ct > 0):
                            k_label = rgb2label(p[k])
                            root[k_label] += 1

                            if (visit[k] == 0):
                                if rgb2label(p[k]) == rgb2label(p[i]):
                                    temp.append(point[k])
                                    all.append(k)
                                    visit[k] = 1
                        ct += 1
                queue = temp

            for j in range(len(all)):
                # if len(all) < 3:
                #     able[all[j]] = 0
                label = np.argwhere(root == np.amax(root)).flatten().tolist()
                if len(label) == 1:
                    r_new[all[j]] = label_colours[label[0]][2]
                    g_new[all[j]] = label_colours[label[0]][1]
                    b_new[all[j]] = label_colours[label[0]][0]
                else: # 存在多个最大值
                    l = np.argmax(p[all[j]])
                    r_new[all[j]] = label_colours[l][2]
                    g_new[all[j]] = label_colours[l][1]
                    b_new[all[j]] = label_colours[l][0]

    logging.info("refine finished")
    print('refine finished')

    result_xyz = []
    result_rgb = []
    print('able:', len(able))
    for m in range(len(able)):
        if able[m]:
            result_xyz.append([x[m] , y[m], z[m]])
            result_rgb.append([r_new[m], g_new[m], b_new[m]])
    return result_xyz, result_rgb
###################### main ############################

# 主函数，通过循环分批读取稠密点云，避免内存爆炸
for ii in range(batch):
    print(TYPE)
    logging.info("iter: {} start".format(ii))
    print("iter: {} start".format(ii))

    # read probability and 2-3D relation
    if TYPE < 2:
        [x, y, z, p] = readTxt(ii, False)
        points_new, p_new = knn_fusion(x, y, z, p)
    elif TYPE == 2:
        [x, y, z, p] = readTxt(ii, True)
        points_new, p_new = knn_fusion(x, y, z, p)
    elif TYPE == 3:
        [x, y, z, p] = readPointCloud(ii)
        points_new, p_new = energy_fusion(x, y, z, p)


    # write Point Cloud
    if TYPE == 0:
        writePointCloud(points_new, p_new, path + "semantic/scene_dense_baseline.obj")
    elif TYPE == 1:
        writePointCloud(points_new, p_new, path + "semantic/scene_dense_simple_k=" + str(K) + "batch_size=" + str(batch) +".obj")
    elif TYPE == 2:
        writePointCloud(points_new, p_new, path + "semantic/scene_dense_softmax_k=" + str(K) + "batch_size=" + str(batch) +".obj")
    else:
        writePointCloud(points_new, p_new, path + "semantic/scene_dense_graph_k=" + str(K) + "batch_size=" + str(batch) + ".obj")

    logging.info("write finished")
    print("write finished")
