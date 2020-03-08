from util import writePointCloud
old_colormap = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(0,0,0)] #BGR sequence
new_colormap = [(35,142,107),(70,70,70),(128,64,128),(142,0,0),(0,0,0)] #BGR sequence

path="/data1/Dataset/game/Tianjin_semantic/"
filepath = "semantic/scene_dense_softmax_k=4.obj"
file1 = open(path + filepath)

xyz = []
p = []
line = file1.readline()
while line:
    dt = line.split()
    xyz.append([float(dt[1]), float(dt[2]), float(dt[3])])
    flag = 0
    for i in range(len(old_colormap)):
        if int(dt[6]) == old_colormap[i][0] and int(dt[5]) == old_colormap[i][1] and int(dt[4]) == old_colormap[i][2]:
            flag = 1
            p.append(new_colormap[i])
    if flag == 0:
        p.append(old_colormap[i])
    line = file1.readline()

file1.close()

writePointCloud(xyz, p, path + "semantic/change_scene_dense_softmax_k=4.obj")
