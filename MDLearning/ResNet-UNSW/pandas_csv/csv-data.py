
import pandas as pd
#获取特征名列表作为列名
from columns import columns

dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df = pd.read_csv("../UNSW_datas/UNSW-NB15_4.csv", names=featurelist)

#设置显示格式
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)
#df= df.replace('-',0)
# df2 = df.service.value_counts()
# df3 = df.proto.value_counts()


#获取df.service,df.proto所有列名
lie = df.columns.values.tolist()
print(lie)
print('====')
#通过列名列表获取这一列
# for i in range(len(lie)):
#     print('第:',i,'列  ',lie[i])
#     print(df[lie[i]].value_counts())





#UNSW-NB15_4.csv清洗工作
print('清洗·============================================================')
print('将攻击类型补全')
df.loc[df['Label']==0, 'attack_cat'] = 'Normal'
# # 比如，把Label中值为的行对应的attack_cat替换成'Normal'

lie = df.columns.values.tolist()
for i in range(len(lie)):
    print('第:',i,'列  ',lie[i])
    print(df[lie[i]].value_counts())
print('根据态势指标将类型转换成对应数字')
df= df.replace('Normal',0)
df= df.replace('Analysis',1)
df= df.replace(' Reconnaissance ',2)
df= df.replace('DoS',3)
df= df.replace(' Fuzzers ',4)
df= df.replace('Generic',5)
df= df.replace(' Shellcode ',6)
df= df.replace('Worms',7)
df= df.replace('Exploits',8)
df= df.replace('Backdoor',9)
# print(df)


print('删除原地址和目的地址ip列')
df = df.drop(labels=['srcip','dstip'],axis=1)
# print(df)


print('将service列的-换成default')
df = df.replace('-','default')
# print(df)

print('将空NaN的-换成0')
df = df.fillna(0)
print('将' '-换成0')
df = df.replace(' ',0)
# print(df)
print('删除Label列')
df = df.drop(labels=['Label'],axis=1)

print('#通过列名列表获取这一列')
lie = df.columns.values.tolist()
print('df long:',len(lie))
print(lie)


print('独热编码=======================')
#get_dummies是pandas方法
dummies1 = pd.get_dummies(df['proto'],prefix='proto')

#获取所有列名
l1 = dummies1.columns.values.tolist()
print(l1)
print('dumies1 long:',len(l1))#这是proto的种类


dummies2 = pd.get_dummies(df['state'],prefix='state')
l2 = dummies2.columns.values.tolist()
print(l2)
print('dumies2 long:',len(l2))

dummies3 = pd.get_dummies(df['service'],prefix='service')
l3 = dummies3.columns.values.tolist()
print(l3)
print('dumies3 long:',len(l3))

lie2 = df.columns.values.tolist()
print(lie2)
print('df long:',len(lie2))


print('拼接=======================')

df1=df[['sport', 'dsport']]
lie3 = df1.columns.values.tolist()
print('df1 long:',len(lie3))

df2=df[lie2[4:11]]
lie4 = df2.columns.values.tolist()
print('df2 long:',len(lie4))

df3=df[lie2[12:]]
lie5 = df3.columns.values.tolist()
print('df3 long:',len(lie5))

#按照列合并
df=pd.concat([df1,dummies1,dummies2,df2,dummies3 ,df3],axis=1,ignore_index=False)
lietotal = df.columns.values.tolist()
print('lietotal long:',len(lietotal))
print(lietotal)
# print(df)


#扩张维度
label_df = df[['attack_cat']]  #获取标签
df = df.drop(labels=['attack_cat'],axis=1) #删除最后一列
li = ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13']
for i in range(13):
    df.loc[:, li[i]] = 0   #加一列，所有行赋值
lietotal = df.columns.values.tolist()
print(lietotal)
#df=pd.concat([df,label_df],axis=1,ignore_index=False)    标准化后再拼接

print('标准化=======================')


print('finifished')
