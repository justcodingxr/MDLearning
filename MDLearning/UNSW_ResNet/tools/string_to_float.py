import pandas as pd
#获取特征名列表作为列名
from columns import columns

dffeature = pd.read_csv("../pandas_csv/get-tvt-train-val-test-data/feature.csv")
featurelist = dffeature['Name']
df1 = pd.read_csv("../UNSW_datas/UNSW-NB15_1.csv", names=featurelist, low_memory=False)
df2 = pd.read_csv("../UNSW_datas/UNSW-NB15_2.csv", names=featurelist, low_memory=False)
df3 = pd.read_csv("../UNSW_datas/UNSW-NB15_3.csv", names=featurelist, low_memory=False)
df4 = pd.read_csv("../UNSW_datas/UNSW-NB15_4.csv", names=featurelist, low_memory=False)

#设置显示格式
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)


# print('df1:',df1.dtypes)
df1 = df1.drop(labels=['srcip','dstip'],axis=1)
# print('df1:',df1.dtypes)
# print('df2:',df2.dtypes)
# print('df3:',df3.dtypes)
# print('df4:',df4.dtypes)

# df1.to_csv('dfile1.csv',index=False,header=False,encoding="utf_8_sig")
# df11 = pd.read_csv("dfile1.csv", encoding='utf-8',names=featurelist, low_memory=False)
# print('df11:',df11.dtypes)

dummies2 = pd.get_dummies(df1['proto'],prefix='proto')
# print('dumies2:',dummies2.dtypes)
l2 = dummies2.columns.values.tolist()
# print(l2)
# print('dumies2 long:',len(l2))#这是proto的种类
lie1 = df1.columns.values.tolist()
# print(lie1)
# print('df long:',len(lie1))


p1=df1[lie1[0:2]]
# print('p1:',p1.dtypes)

p1d1=pd.concat([p1,dummies2],axis=1,ignore_index=False)
# print(p1d1.dtypes)


df1.loc[df1['Label']==0, 'attack_cat'] = 'Normal'
att1= df1['attack_cat']
# print('att1:',att1.dtypes)

p1d1a1=pd.concat([p1d1,att1],axis=1,ignore_index=False)
# print('p1d1a1:',p1d1a1.dtypes)

# print(p1d1a1)

# p1d1a1= p1d1a1.replace('Normal','Normal')
# p1d1a1= p1d1a1.replace('Analysis','Analysis')
# p1d1a1= p1d1a1.replace(' Reconnaissance ','Reconnaissance')
# p1d1a1= p1d1a1.replace('Reconnaissance','Reconnaissance')
# p1d1a1= p1d1a1.replace('DoS','DoS')
# p1d1a1= p1d1a1.replace(' Fuzzers ','Fuzzers')
# p1d1a1= p1d1a1.replace(' Fuzzers','Fuzzers')
# p1d1a1= p1d1a1.replace('Generic','Generic')
# p1d1a1= p1d1a1.replace(' Shellcode ','Shellcode')
# p1d1a1= p1d1a1.replace('Shellcode','Shellcode')
# p1d1a1= p1d1a1.replace('Worms','Worms')
# p1d1a1= p1d1a1.replace('Exploits','Exploits')
# p1d1a1= p1d1a1.replace('Backdoor','Backdoors')
# p1d1a1= p1d1a1.replace('Backdoors','Backdoors')
#
# p1d1a1= p1d1a1.replace('Normal',0)
# p1d1a1= p1d1a1.replace('Analysis',1)
# p1d1a1= p1d1a1.replace('Reconnaissance',2)
# p1d1a1= p1d1a1.replace('Fuzzers',4)
# p1d1a1= p1d1a1.replace('DoS',3)
# p1d1a1= p1d1a1.replace('Generic',5)
# p1d1a1= p1d1a1.replace('Shellcode',6)
# p1d1a1= p1d1a1.replace('Worms',7)
# p1d1a1= p1d1a1.replace('Exploits',8)
# p1d1a1= p1d1a1.replace('Backdoors',9)
print('p1d1a1:',p1d1a1.dtypes)


p1d1a1 = p1d1a1.apply(pd.to_numeric,errors ='coerce')
print('string to p1d1a1:',p1d1a1.dtypes)
print('p1d1a1:')

# #设置显示格式
# pd.set_option('display.max_rows', 2000)
# pd.set_option('display.max_columns', 200)
# pd.set_option('display.width', 160)
# print(p1d1a1)
# print(p1d1a1['dsport'])
# print(p1d1a1['attack_cat'])
# print(p1d1a1['proto_3pc'])
#
# lie = p1d1a1.columns.values.tolist()
# for i in range(len(lie)):
#     print('第:',i,'列  ',lie[i])
#     print(p1d1a1[lie[i]].value_counts())
