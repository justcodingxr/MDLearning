import pandas as pd
#获取特征名列表作为列名
from columns import columns

dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df = pd.read_csv("../../UNSW_datas/UNSW-NB15_1.csv", names=featurelist, low_memory=False)

#设置显示格式
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)
#df= df.replace('-',0)
# df2 = df.service.value_counts()
# df3 = df.proto.value_counts()

#获取df.service,df.proto所有列名
lie = df.columns.values.tolist()
print(len(lie))
print(lie)

#通过列名列表获取这一列
# for i in range(len(lie)):
#     print('第:',i,'列  ',lie[i])
#     print(df[lie[i]].value_counts())


print('将攻击类型补全')
df.loc[df['Label']==0, 'attack_cat'] = 'Normal'

lie = df.columns.values.tolist()
for i in range(len(lie)):
    print('第:',i,'列  ',lie[i])
    print(df[lie[i]].value_counts())
print(df)

print('根据态势指标将类型转换成对应数字')
df= df.replace('Normal',0)
df= df.replace('Analysis',1)
df= df.replace(' Reconnaissance ',2)
df= df.replace('Reconnaissance',2)
df= df.replace('DoS',3)
df= df.replace(' Fuzzers ',4)
df= df.replace(' Fuzzers',4)
df= df.replace('Generic',5)
df= df.replace(' Shellcode ',6)
df= df.replace('Shellcode',6)
df= df.replace('Worms',7)
df= df.replace('Exploits',8)
df= df.replace('Backdoor',9)
df= df.replace('Backdoors',9)
lie = df.columns.values.tolist()
for i in range(len(lie)):
    print('第:',i,'列  ',lie[i])
    print(df[lie[i]].value_counts())