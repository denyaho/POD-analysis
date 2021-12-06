import math
import numpy as np
from numpy import linalg as LA
import glob
import os
import csv

final = []

with open("./datapod/velocity_00000100.dat","r") as f:
    uu1,vv1,ww1 = np.loadtxt(f,unpack = True)
    uv1 = np.hstack([uu1,vv1,ww1])
#データを横につなげる
#データを縦につなげる
length = len(uu1)
duallength = length*2

with open("./datapod/velocity_00000200.dat","r") as f:
    uu2,vv2,ww2 = np.loadtxt(f,unpack = True)
    uv2 = np.hstack([uu2,vv2,ww2])

with open("./datapod/velocity_00000300.dat","r") as f:
    uu3,vv3,ww3 = np.loadtxt(f,unpack = True)
    uv3 = np.hstack([uu3,vv3,ww3])

with open("./datapod/velocity_00000400.dat","r") as f:
    uu4,vv4,ww4 = np.loadtxt(f,unpack = True)
    uv4 = np.hstack([uu4,vv4,ww4])

with open("./datapod/velocity_00000500.dat","r") as f:
    uu5,vv5,ww5 = np.loadtxt(f,unpack = True)
    uv5 = np.hstack([uu5,vv5,ww5])

with open("./datapod/velocity_00000600.dat","r") as f:
    uu6,vv6,ww6 = np.loadtxt(f,unpack = True)
    uv6 = np.hstack([uu6,vv6,ww6])

with open("./datapod/velocity_00000700.dat","r") as f:
    uu7,vv7,ww7 = np.loadtxt(f,unpack = True)
    uv7 = np.hstack([uu7,vv7,ww7])

with open("./datapod/velocity_00000800.dat","r") as f:
    uu8,vv8,ww8 = np.loadtxt(f,unpack = True)
    uv8 = np.hstack([uu8,vv8,ww8])

with open("./datapod/velocity_00000900.dat","r") as f:
    uu9,vv9,ww9 = np.loadtxt(f,unpack = True)
    uv9 = np.hstack([uu9,vv9,ww9])

with open("./datapod/velocity_00001000.dat","r") as f:
    uu10,vv10,ww10 = np.loadtxt(f,unpack = True)
    uv10 = np.hstack([uu10,vv10,ww10])

comb = np.vstack([uv1,uv2,uv3,uv4,uv5,uv6,uv7,uv8,uv9,uv10])

#uvwの転置
transpose = np.transpose(comb)

#配列の掛け算
square = np.dot(comb,transpose)
#固有値、固有ベクトルの抽出(固有値w、固有ベクトルv)
value, vector = LA.eig(square)
eigen_id = value.argsort()[::-1]
value = value[eigen_id]
vector = vector[eigen_id]
with open('eigen.dat','w') as e:
    print(value,vector,file=e)


#base vectorの算出
base0 = np.dot(transpose,vector[0])/np.sqrt(value[0])
base1 = np.dot(transpose,vector[1])/np.sqrt(value[1])
base2 = np.dot(transpose,vector[2])/np.sqrt(value[2])
base3 = np.dot(transpose,vector[3])/np.sqrt(value[3])
base4 = np.dot(transpose,vector[4])/np.sqrt(value[4])
base5 = np.dot(transpose,vector[5])/np.sqrt(value[5])
base6 = np.dot(transpose,vector[6])/np.sqrt(value[6])
base7 = np.dot(transpose,vector[7])/np.sqrt(value[7])
base8 = np.dot(transpose,vector[8])/np.sqrt(value[8])
base9 = np.dot(transpose,vector[9])/np.sqrt(value[9])

#np.set_printoptions(threshold=np.inf)
#test1 = [base0[i] for i in range(0,length)]
#test2 = [base0[i] for i in range(length,duallength)]
with open("test1.dat","w") as f:
    for i in range(0,length):
        print('1.0 {0} {0} {0} 0.20543667623446646'.format(base1[i],base1[i+length],base1[i+duallength]),file=f)
