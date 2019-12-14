# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np
import argparse
import logging
import os
import operator
from util.util import calcDistance, writePointCloud

###################### params ############################
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, default=0, help='algorithms, 0 for baseline, 1 for simple knn, 2 for softmax knn, 3 for energy min.')
parser.add_argument('--classes', type=int, default=5, help='semantic classes')
parser.add_argument('--K', type=int, default=8, help='nearest neighbors')
parser.add_argument('--batch_size', type=int, default=100, help='divided into n batchs')
parser.add_argument('--resolution_level', type=int, default=1, help='MVS resolution level')
args = parser.parse_args()

label_colours = [(35,142,107),(70,70,70),(128,64,128),(0,0,142),(0,0,0)] # BGR sequence, # 0=vegetarian, 1=building, 2=road 3=vehicle, 4=other
TYPE = args.type
CLASS = args.classes
K = args.K
batch = args.batch_size
resolution = args.resolution_level
POINT_N = 0

# path="/data1/Dataset/knn/"
# path="/data1/Dataset/pku/library/"
path="/data1/Dataset/pku/m1_semantic/"
# path="/data1/Dataset/game/Tianjin_semantic/"

###################### log system setup ############################
logName=""
if TYPE == 0:
    logName = "{}baseline_resolution={}.txt".format(path, resolution)
elif TYPE == 1:
    logName = "{}argmax_knn_resolution={}_k={}.txt".format(path, resolution, K)
elif TYPE == 2:
    logName = "{}prob_knn_resolution={}_k={}.txt".format(path, resolution, K)
else:
    logName = "{}energy_resolution={}.txt".format(path, resolution)

logging.basicConfig(filename=logName, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Algorithm begin: path={}, logName={}, batch={}".format(path, logName, batch,))
print("Algorithm begin: path={}, logName={}, batch={}".format(path, logName, batch))

###################### dataset setup ############################

# KNN test
# fx=[5,4,3,2,7,0,9,8,1,6]
# WIDTH = 4000 / resolution
# HEIGHT = 3000 / resolution
# prediction = [0]*10
# for a in range(10):
#     prediction[a] = np.load(path + "prediction/" + str(a) + ".npy")

# PKU library
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

# PKU m1
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

# Contest Tianjin
# fx=[ 783, 784, 782, 780, 781, 779, 777, 778, 776, 774, 775, 773, 771, 772, 770, 768, 769, 767, 765, 766, 764, 762, 763, 761,
#   759, 760, 758, 756, 757, 755, 753, 754, 752, 750, 751, 749, 747, 748, 746, 737, 745, 743, 741, 742, 740, 739, 738, 736, 744,
#   735, 734, 733, 731, 732, 730, 728, 729, 727, 725, 726, 724, 722, 723, 721, 719, 720, 718, 716, 717, 715, 713, 711, 712, 710,
#   708, 709, 707, 705, 706, 704, 702, 703, 701, 699, 700, 698, 696, 690, 695, 694, 693, 691, 692, 697, 689, 688, 686, 687, 685,
#   683, 684, 682, 680, 681, 679, 677, 678, 676, 674, 675, 673, 671, 672, 670, 668, 669, 667, 665, 666, 664, 655, 663, 661, 660,
#   659, 657, 658, 656, 653, 662, 654, 652, 651, 650, 648, 649, 647, 646, 645, 644, 643, 642, 641, 639, 640, 638, 637, 636, 614,
#   792, 10,  794, 9,   796, 11,  795, 13,  797, 12,  799, 14,  798, 16,  800, 15,  802, 17,  801, 19,  803, 18,  805, 20,  806,
#   21,  804, 22,  807, 24,  809, 23,  808, 25,  810, 27,  811, 26,  812, 28,  814, 30,  813, 29,  815, 31,  817, 33,  816, 32,
#   818, 34,  820, 36,  819, 35,  821, 37,  822, 39,  823, 38,  825, 40,  826, 42,  824, 41,  827, 43,  829, 45,  828, 44,  837,
#   836, 838, 840, 839, 504, 843, 505, 842, 507, 844, 506, 846, 508, 845, 510, 847, 509, 849, 511, 848, 513, 850, 512, 852, 514,
#   851, 516, 853, 515, 855, 517, 854, 519, 856, 518, 858, 520, 857, 522, 859, 521, 860, 523, 861, 525, 863, 524, 862, 526, 864,
#   528, 866, 527, 865, 529, 867, 530, 868, 531, 869, 533, 912, 575, 914, 576, 913, 578, 915, 577, 917, 579, 916, 581, 918, 580,
#   919, 582, 920, 584, 921, 583, 922, 585, 924, 587, 923, 586, 925, 588, 927, 590, 926, 589, 928, 591, 930, 593, 931, 592, 597,
#   933, 929, 594, 608, 946, 943, 612, 610, 945, 942, 607, 609, 947, 944, 606, 611, 605, 620, 941, 603, 939, 604, 940, 602, 938,
#   601, 936, 600, 937, 598, 935, 599, 934, 595, 932, 596, 910, 574, 911, 572, 909, 573, 907, 571, 908, 569, 906, 570, 904, 568,
#   905, 566, 903, 567, 901, 565, 902, 563, 900, 564, 898, 562, 899, 560, 897, 561, 895, 559, 896, 557, 894, 558, 892, 556, 893,
#   554, 891, 555, 889, 553, 890, 551, 888, 552, 886, 550, 887, 548, 885, 549, 883, 547, 884, 545, 882, 546, 880, 544, 881, 542,
#   879, 543, 877, 541, 878, 539, 876, 540, 875, 538, 874, 536, 872, 537, 873, 535, 871, 532, 870, 534, 613, 6,   789, 618, 5,
#   790, 619, 4,   788, 617, 1,   786, 616, 2,   785, 615, 3,   787, 622, 7,   791, 8,   793,  46,  47, 503, 621, 623, 625, 624,
#   626, 628, 629, 627, 630, 631, 632, 633, 634, 635,
# ]
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

###################### functions ############################

# 从ply读点云
def readPointCloud(ii):
    file1 = open(path + "semantic/model_dense.ply")

    for _ in range(13):
        line = file1.readline()

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
            x.append(float(dt[0]))
            y.append(float(dt[1]))
            z.append(float(dt[2]))
            p.append([int(dt[3]) / 255., int(dt[4]) / 255., int(dt[5]) / 255.])

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
                    if (w >= WIDTH) or (h >= HEIGHT) or type(prediction[fx[int(imageInfo[0])]]) == int:
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
                    for m in range(len(root)):
                        root[m] += p[j][m]
                    # use the e distance
                    distance = calcDistance(point[i], point[j])
                    if (distance > 1):
                        root += p[j] / (distance * distance)

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

def energy_fusion(x, y, z, p):
    point = np.array([x, y, z]).transpose()
    POINT_N = point.shape[0]
    logging.info("points: {}".format(POINT_N))
    print("points: {}".format(POINT_N))

    tree = spatial.KDTree(point)
    logging.info("build tree finished")
    print('build tree finished')

    dsum = 0
    k = int(POINT_N / 1000)
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
            root = p[i]

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
                            for m in range(len(root)):
                                root[m] += p[k][m]

                            if (visit[k] == 0):
                                if p[k][0] == p[i][0] and p[k][1] == p[i][1] and p[k][2] == p[i][2]:
                                    temp.append(point[k])
                                    all.append(k)
                                    visit[k] = 1
                        ct += 1
                queue = temp

            for j in range(len(all)):
                if len(all) < 5:
                    able[all[j]] = 0
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
    for m in range(len(able)):
        if able[m]:
            result_xyz.append([x[m] , y[m], z[m]])
            result_rgb.append([r_new[m], g_new[m], b_new[m]])
    return result_xyz, result_rgb
###################### main ############################

# 主函数，通过循环分批读取稠密点云，避免内存爆炸
for ii in range(1):
    print(TYPE)
    logging.info("iter: {} start".format(ii))
    print("iter: {} start".format(ii))

    # read probability and 2-3D relation
    if TYPE < 2:
        [x, y, z, p] = readTxt(ii, False)
    else:
        [x, y, z, p] = readTxt(ii, True)

    # refine start
    if TYPE == 3:
        points_new, p_new = energy_fusion(x, y, z, p)
    else:
        points_new, p_new = knn_fusion(x, y, z, p)

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