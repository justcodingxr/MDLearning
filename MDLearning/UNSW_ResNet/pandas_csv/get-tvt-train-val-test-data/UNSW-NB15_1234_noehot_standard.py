import pandas as pd
#获取特征名列表作为列名
from columns import columns

dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df1 = pd.read_csv("../../UNSW_datas/UNSW-NB15_1.csv", names=featurelist, low_memory=False)
df2 = pd.read_csv("../../UNSW_datas/UNSW-NB15_2.csv", names=featurelist, low_memory=False)
df3 = pd.read_csv("../../UNSW_datas/UNSW-NB15_3.csv", names=featurelist, low_memory=False)
df4 = pd.read_csv("../../UNSW_datas/UNSW-NB15_4.csv", names=featurelist, low_memory=False)

df = pd.concat([df1,df2,df3,df4],axis=0,ignore_index=False)
df = df.sample(frac=1).reset_index(drop=True)

#设置显示格式
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

# lie = df.columns.values.tolist()
# print(len(lie))
# print(lie)

print('将攻击类型补全')
df.loc[df['Label']==0, 'attack_cat'] = 'Normal'
# # 比如，把Label中值为的行对应的attack_cat替换成'Normal'

#标签一致
df= df.replace('Normal','Normal')
df= df.replace('Analysis','Analysis')
df= df.replace(' Reconnaissance ','Reconnaissance')
df= df.replace('Reconnaissance','Reconnaissance')
df= df.replace('DoS','DoS')
df= df.replace(' Fuzzers ','Fuzzers')
df= df.replace(' Fuzzers','Fuzzers')
df= df.replace('Generic','Generic')
df= df.replace(' Shellcode ','Shellcode')
df= df.replace('Shellcode','Shellcode')
df= df.replace('Worms','Worms')
df= df.replace('Exploits','Exploits')
df= df.replace('Backdoor','Backdoors')
df= df.replace('Backdoors','Backdoors')
df = df.sample(frac=1).reset_index(drop=True)
# df.to_csv('UNSW-NB15_1234.csv',index=False,encoding="utf_8_sig")
# lie = df.columns.values.tolist()
# for i in range(len(lie)):
#     print('第:',i,'列  ',lie[i])
#     print(df[lie[i]].value_counts())


#先清晰，转换一些特俗字符再编码
print('根据态势指标将类型转换成对应数字')
df= df.replace('Normal',0)
df= df.replace('Analysis',1)
df= df.replace('Reconnaissance',2)
df= df.replace('DoS',3)
df= df.replace('Fuzzers',4)
df= df.replace('Generic',5)
df= df.replace('Shellcode',6)
df= df.replace('Worms',7)
df= df.replace('Exploits',8)
df= df.replace('Backdoors',9)

print('删除原地址和目的地址ip列')
df = df.drop(labels=['srcip','dstip'],axis=1)


print('将service列的-换成default')
df = df.replace('-','defvice')
# print(df)

print('将空NaN的-换成0')
df = df.fillna(0)
print('将' '-换成0')
df = df.replace(' ',0)
df = df.replace('',0)
df = df.replace('  ',0)
df = df.replace('   ',0)
df = df.replace('    ',0)
# print(df)
print('删除Label列')
df = df.drop(labels=['Label'],axis=1)
lie = df.columns.values.tolist()
print(len(lie))
# print(lie)

#one-hot编码，先编码在抽取简化，防止维度变化
print('独热编码=======================')
print('参照列：')
lie = df.columns.values.tolist()
print('df long:',len(lie))
print(lie)


dummies1 = pd.get_dummies(df['proto'],prefix='proto')
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

print('拼接================================================================')

df1=df[lie2[0:2]]
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


#简化数据集,Normal,Generic 各自抽取100000，与剩余的构成新的数据集。
df_normal = df.loc[df['attack_cat']==0]  #[215481 rows x 49 columns]
df_generic = df.loc[df['attack_cat']==5] #[2218764 rows x 49 columns]
df_other =  df.loc[df['attack_cat'].isin([1,2,4,3,6,7,8,9])]
df_normal_part =df_normal.sample(n=100000,  replace=False, weights=None, random_state=None, axis=0)
df_generic_part =df_generic.sample(n=100000,  replace=False, weights=None, random_state=None, axis=0)

df=pd.concat([df_normal_part,df_generic_part,df_other],axis=0,ignore_index=False)
print(df)
df = df.sample(frac=1).reset_index(drop=True)

print('扩张维度：206+19+1')
label_df = df[['attack_cat']]  #获取标签
df = df.drop(labels=['attack_cat'],axis=1) #删除最后一列
li = ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18']
for i in range(19):
    df.loc[:, li[i]] = 0   #加一列，所有行赋值
lietotal = df.columns.values.tolist()   #不含attacka_cat
print(len(lietotal))
print(lietotal)
print(df.dtypes)
print(df.dtypes)
# df['dsport'] = df['dsport'].apply(pd.to_numeric,errors ='coerce')           #string对象转为float
# df.to_csv('UNSW-NB15_1234_after-onehot.csv',index=False,header=False,encoding="utf_8_sig")


print('string to float NaN to 0==============================================')
print('只转换能转换的，不能的转换为NaN')
df = df.apply(pd.to_numeric,errors ='coerce')
df = df.fillna(0)



print('标准化=======================================================================')
import numpy as np
from  sklearn import preprocessing
scaler=preprocessing.StandardScaler()
#df=scaler.fit_transform(np.array(df))
df=scaler.fit_transform(df) #已经变成了一个array数组
df = pd.DataFrame(df,columns=lietotal)              #传入list作为列名，featurelist是一列dataframe
print(df)

#拼接attack_cat
df=pd.concat([df,label_df],axis=1,ignore_index=False)

print(df.dtypes)
for i in range(200):
    #必须足够乱，其中由于打乱后的数据，index会乱掉，需要重置index
    # 在重置index后，旧的index默认会成为数据中的一列，因此设置参数drop=True，表示删掉旧的index
    df = df.sample(frac=1).reset_index(drop=True)
print(df.dtypes)



#分割
df_train = df[0:240000][:]   #[240000 rows x 49 columns]
df_val = df[240000:270000][:]   #[30000 rows x 49 columns]
df_test = df[270000:305802][:]        #[35802 rows x 49 columns]

print('tarin:',len(df_train))
print(df_train.shape[0])
print(df_train.shape[1])

print('val:',len(df_val))
print(df_val.shape[0])
print(df_val.shape[1])

print('test:',len(df_test))
print(df_test.shape[0])
print(df_test.shape[1])


#多次运行选择分布较好的
lie1 = df_train.columns.values.tolist()
for i in range(225,226):
    print('train:',df_train[lie1[i]].value_counts())

lie2 = df_val.columns.values.tolist()
for i in range(225,226):
    print('val:',df_val[lie2[i]].value_counts())

lie3 = df_test.columns.values.tolist()
for i in range(225,226):
    print('test:',df_test[lie3[i]].value_counts())



#保存分割好的数据集
df_train.to_csv('UNSW-NB15_1234_simple_train_after-onehot-standard.csv',index=False,header=False)
df_val.to_csv('UNSW-NB15_1234_simple_val_after-onehot-standard.csv',index=False,header=False)
df_test.to_csv('UNSW-NB15_1234_simple_test_after-onehot-standard.csv',index=False,header=False)



