import pandas as pd
#获取特征名列表作为列名
from columns import columns

dffeature = pd.read_csv("feature.csv")

featurelist = dffeature['Name']
print('list:',featurelist)
print('dffeature:',dffeature)

df = pd.read_csv("../UNSW_beforeclean/UNSW-23to1-train-raw.csv", names=featurelist, low_memory=False)

#设置显示格式
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

print('清洗·============================================================')
print('将攻击类型补全')
df.loc[df['Label']==0, 'attack_cat'] = 'Normal'
# 比如，把Label中值为的行对应的attack_cat替换成'Normal'
lie = df.columns.values.tolist()
print(lie)
# 通过列名列表获取这一列
# for i in range(len(lie)):
#     print('第:',i,'列  ',lie[i])
#     print(df[lie[i]].value_counts())
# Normal            10000
# Generic            7522
# Exploits           5409
# Fuzzers            5051
# Analysis           2007
# Reconnaissance     1759
# DoS                1167
# Backdoor           1663
# Shellcode          1140
# Worms               131
# Name: attack_cat, dtype: int64
#[35849 rows x 49 columns]
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


df = df.drop(labels=['srcip','dstip'],axis=1)
df = df.replace('-','default')
df = df.fillna(0)
df = df.replace(' ',0)
df = df.drop(labels=['Label'],axis=1)
lie = df.columns.values.tolist()
print('df long:',len(lie))
print(lie)

print('独热编码·============================================================')
dummies1 = pd.get_dummies(df['proto'],prefix='proto')
l1 = dummies1.columns.values.tolist()
print('dumies1 long:',len(l1))#这是proto的种类
print(l1)

dummies2 = pd.get_dummies(df['state'],prefix='state')
l2 = dummies2.columns.values.tolist()
print('dumies2 long:',len(l2))
print(l2)

dummies3 = pd.get_dummies(df['service'],prefix='service')
l3 = dummies3.columns.values.tolist()
print('dumies3 long:',len(l3))
print(l3)

lie2 = df.columns.values.tolist()
print('df long:',len(lie2))
print(lie2)


print('拼接================================================================')

df1=df[['sport', 'dsport']]
lie3 = df1.columns.values.tolist()
print('df1 long:',len(lie3))
print(lie3)

df2=df[lie2[4:11]]
lie4 = df2.columns.values.tolist()
print('df2 long:',len(lie4))
print(lie4)

df3=df[lie2[12:]]
lie5 = df3.columns.values.tolist()
print('df3 long:',len(lie5))
print(lie5)

#按照列合并
df=pd.concat([df1,dummies1,dummies2,df2,dummies3 ,df3],axis=1,ignore_index=False)
lietotal = df.columns.values.tolist()
print('lietotal long:',len(lietotal))
print(lietotal)

#扩张维度
label_df = df[['attack_cat']]  #获取标签
df = df.drop(labels=['attack_cat'],axis=1) #删除最后一列
li = ['d0']
df.loc[:, li[0]] = 0   #加一列，所有行赋值
lietotal = df.columns.values.tolist()
print(lietotal)
#df=pd.concat([df,label_df],axis=1,ignore_index=False)    #标准化后再拼接
lietotal = df.columns.values.tolist()
print('lietotal long:',len(lietotal))
print(lietotal)
print('标准化=======================================================================')
#标准化
from  sklearn import preprocessing
scaler=preprocessing.StandardScaler()
df=scaler.fit_transform(df)                  #已经变成了一个array数组
df = pd.DataFrame(df,columns=lietotal)              #传入list作为列名，featurelist是一列dataframe


#拼接
df=pd.concat([df,label_df],axis=1,ignore_index=False)

#随机行顺序,多做几次
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('../UNSW_afterclean/UNSW_train.csv',index=False,header=False,encoding="utf_8_sig")