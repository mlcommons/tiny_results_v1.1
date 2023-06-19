

#!/usr/bin/python3
# -*- coding: UTF-8 -*-





from numpy import *
import numpy as np
np.set_printoptions(threshold=np.inf,linewidth=3000)


data = np.loadtxt('data_conv_bias_pos.txt',dtype=int)
pose=np.array(data)
pose=pose.reshape(28,2)


A = []
f = open('./data_weight_int32.txt')  


lines = f.readlines() 
data_len = 0  
for line in lines:  
    list = line.strip(',\n').split(',')  
    B=len(list)
    A = A + list[0:B]  
    data_len = data_len + 1


A=np.array(A)
D=len(A)
B=A.reshape(D,1)


zero_8=['00']

file=open('./bias_pw.txt','w')  

x=0

for k in range(15):
    if(x>=27):
        x=27
    else:
        x=x

    pose_1 = pose[x][0]
    pose_2 = pose[x][1]
    C=B[pose_1:(pose_1+pose_2),0:1]

    for l in range(pose_2):
        D=C[l:(l+1),0:1]

        F=int(D[0][0],16)
        if(F<0):
            F=hex((F + (1 << 32)) % (1 << 32))
        else:
            F='{:0>8x}'.format(F)

        file.write(str(F))
        file.write(str('\r\n'))

    x=x+2

file.close() 



file=open('./bias_dw.txt','w')  

x=0

for k in range(13):
    y=x+1

    pose_1 = pose[y][0]
    pose_2 = pose[y][1]
    C=B[pose_1:(pose_1+pose_2),0:1]

    for l in range(pose_2):
        D=C[l:(l+1),0:1]

        F=int(D[0][0],16)
        if(F<0):
            F=hex((F + (1 << 32)) % (1 << 32))
        else:
            F='{:0>8x}'.format(F)

        file.write(str(F))
        file.write(str('\r\n'))
    x=x+2
    

file.close() 



















