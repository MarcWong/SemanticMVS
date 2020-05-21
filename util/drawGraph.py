
import matplotlib.pyplot as plt
import numpy as np



fig = plt.figure(figsize=(32,28))
# x = (261, 337, 346, 398, 413, 437, 440, 497, 516, 529, 534, 539, 540, 544, 561, 563, 573, 574, 581, 593, 596, 604, 611, 622, 624)
# dense = (23, 26, 30, 32, 35, 37, 34, 43, 47, 48, 49, 48, 49, 51, 50, 50, 50, 52, 51, 50, 52, 53, 52, 60, 56)
# mesh = (22, 18, 19, 23, 27, 47, 27, 40, 43, 48, 52, 27, 31, 35, 31, 37, 46, 38, 43, 54, 59, 56, 37, 38, 40)
# refine = (120, 120, 143, 155, 205, 190, 162, 286, 313, 371, 384, 311, 335, 313, 351, 345, 348, 446, 330, 321, 318, 436, 365, 420, 421)

x = (18.4, 18.8, 20.6, 21.9, 22.3, 23.6, 28.6, 33.0,  33.6, 34.9, 38.5, 39.3, 39.9,  40.2, 40.6, 42.1, 44.2, 45.2, 47.6, 50.0,51.6, 51.7,  51.9, 56.2)
dense = (26, 30, 32, 34, 35, 23,  48, 49, 50, 52, 37, 43, 51,  52, 51, 47, 60, 56, 50, 48,52, 49, 50, 53)
mesh = (18, 19, 23, 27, 27, 22, 48, 31, 31, 38, 47, 40,  35,  37, 43, 43, 38, 40, 46,  52, 59, 27, 54, 56)
refine = (120, 143, 155, 162, 205, 120, 311, 335, 351, 446, 190, 286, 313,  365, 330,  313, 420, 421, 348, 371, 318, 384, 321,436)

plt.plot(x,dense,label='Dense Cloud Reconstruction')
plt.plot(x,mesh,label='Mesh Reconstrunction')
plt.plot(x,refine,label='Mesh Refinement')

plt.xticks(x,fontsize=12, rotation=90)#设置x的刻度
plt.yticks(fontsize=24)#设置x的刻度
# plt.xticks(x[::2])
plt.ylabel('Time Consuming(Min)',fontsize=36)
# plt.xlabel('Image Number',fontsize=36)
plt.xlabel('3D Points',fontsize=36)

plt.legend(loc=2, ncol=3, fontsize=32)

plt.savefig('MVS_time.png', format='png')
