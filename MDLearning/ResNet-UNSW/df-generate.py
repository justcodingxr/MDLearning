
#生成UNSW模仿数据集
from random import random
import  numpy as np
import  pandas as pd
ar = np.random.randint(0,254,4900000).reshape(100000,49)
print(ar)
df1 = pd.DataFrame(ar)
print(df1)
li=[]
for i in range(100000):
    x = np.random.randint(0, 9)
    li.append(x)
print(len(li))
print(li)
df1[49]=li
print(df1)
df1.to_csv('UNSW_datas/100_49data.csv',index=False,header=False,encoding="utf_8_sig")
print('finifished')