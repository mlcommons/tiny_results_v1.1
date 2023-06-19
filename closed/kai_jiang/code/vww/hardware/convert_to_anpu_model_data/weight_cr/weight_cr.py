

#!/usr/bin/python3
# -*- coding: UTF-8 -*-





from numpy import *
import numpy as np
np.set_printoptions(threshold=np.inf,linewidth=3000)


data = np.loadtxt('data_conv_weight_pos.txt',dtype=int)
pose=np.array(data)
pose=pose.reshape(28,2)


data_inout = np.loadtxt('in_out_len.txt',dtype=int)
pose_inout =np.array(data_inout)
pose_inout =pose_inout.reshape(28,2)

A = []
f = open('./data_weight_int8.txt')  


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

file=open('./weight_pw.txt','w')  

x=0
m_8=0
m_32=0

for k in range(14):
    x=x+2

    if(x>=27):
        y=27
    else:
        y=x


    pose_1 = pose[y][0]
    pose_2 = pose[y][1]
    
    in_len = pose_inout[y][0]
    out_len= pose_inout[y][1]
    
    C=B[pose_1:(pose_1+pose_2),0:1]
    
    

    if(k<=1):
        for l in range(pose_2):

            D=C[l:(l+1),0:1]

            F=int(D[0][0],16)
            if(F<0):
                F=hex((F + (1 << 8)) % (1 << 8))
            else:
                F='{:0>2x}'.format(F)

            file.write(str(F))
            file.write(str('\r\n'))

            l_add=l+1
            if(((l_add-m_8)==8)  and (k==0)):
                m_8=l_add
                for u in range(8):
                    U='{:0>2x}'.format(0)
                    file.write(str(U))
                    file.write(str('\r\n'))
            else:
                m_8=m_8
    elif(k>=2 and k<13):
        CC=C.reshape(out_len,in_len)
        in_1 =int(in_len/16)
        out_1=int(out_len/16)
        
        for out_i in range (out_1):
            dd=CC[out_i*16:(out_i+1)*16,0:in_len]
            for in_i in range (in_1):
                ee=dd[0:16,in_i*16:(in_i+1)*16]
                ee=ee.reshape(256,1)
                for i_256 in range(256):
                    F=int(ee[i_256][0],16)
                    if(F<0):
                        F=hex((F + (1 << 8)) % (1 << 8))
                    else:
                        F='{:0>2x}'.format(F)

                    file.write(str(F))
                    file.write(str('\r\n'))
    else:
        CC=C.reshape(out_len,in_len)
        in_1 =int(in_len/16)

        dd=CC[0:2,0:in_len]
        for in_i in range (in_1):
            ee=dd[0:2,in_i*16:(in_i+1)*16]
            ee=ee.reshape(32,1)
            for i_256 in range(32):
                F=int(ee[i_256][0],16)
                if(F<0):
                    F=hex((F + (1 << 8)) % (1 << 8))
                else:
                    F='{:0>2x}'.format(F)

                file.write(str(F))
                file.write(str('\r\n'))
            for u in range(32):
                U='{:0>2x}'.format(0)
                file.write(str(U))
                file.write(str('\r\n'))





file.close() 



file=open('./weight_dw.txt','w')  

x=0
a_00=0


for k in range(14):
    if(k<2):
        y=k
    else:
        y=x


    pose_1 = pose[y][0]
    pose_2 = pose[y][1]
    C=B[pose_1:(pose_1+pose_2),0:1]




    if(a_00==0):
        if(y==0):
            for a_3 in range(3):
                for a_8 in range(8):
                    A_8=C[a_8*27:(a_8+1)*27,0:1]
                    A_3=A_8[a_3*9 :(a_3+1)*9 ,0:1]
                    for a_9 in range(9):
                        A_9=A_3[a_9:(a_9+1),0:1]
                        F=int(A_9[0][0],16)
                        if(F<0):
                            F=hex((F + (1 << 8)) % (1 << 8))
                        else:
                            F='{:0>2x}'.format(F)
                        file.write(str(F))
                        file.write(str('\r\n'))
        else:
            for l in range(pose_2):
                D=C[l:(l+1),0:1]
                F=int(D[0][0],16)
                if(F<0):
                    F=hex((F + (1 << 8)) % (1 << 8))
                else:
                    F='{:0>2x}'.format(F)

                file.write(str(F))
                file.write(str('\r\n'))
    else:
        y=y

    if(k==0):
        x=x+1
    else:
        x=x+2

file.close() 



















