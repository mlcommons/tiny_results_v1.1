

#!/usr/bin/python3
# -*- coding: UTF-8 -*-





from numpy import *
import numpy as np
np.set_printoptions(threshold=np.inf,linewidth=3000)

A = []
f = open('./data_quantization_factors.txt')  


lines = f.readlines() 
data_len = 0  
for line in lines:  
    list = line.strip(',\n').split(',')  
    B=len(list)
    A = A + list[0:B]  
    data_len = data_len + 1



A=np.array(A)
B=A.reshape(28,8)

zero_8=['00']

file=open('./qf_003.txt','w')  

for k in range(28):
    if(k==27):
        C_1=B[(k+1):(k+2),0:8]
    else:
        C_1=B[(k+1):(k+2),0:8]
    C=B[k:(k+1),0:8]

    if(k==27):
        E=int('0',10)
    else:
        E=int(C_1[0][0],10)
    E='{:0>8x}'.format(E)
    file.write(E)
    
    E=int(C[0][4],10)
    E='{:0>8x}'.format(E)
    file.write(E)

    if(k==27):
        E=int('0',10)
    else:
        E=int(C_1[0][1],10)
    E='{:0>2x}'.format(E)
    file.write(E)

    E=int(C[0][5],10)
    E='{:0>2x}'.format(E)
    file.write(E)

    file.write('000000000000')
    file.write(str('\r\n'))



file.close() 






















