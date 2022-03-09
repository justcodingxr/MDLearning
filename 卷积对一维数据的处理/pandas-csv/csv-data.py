import pandas as pd
#获取特征名列表作为列名
dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df = pd.read_csv("../UNSW-datas/UNSW-NB15_4.csv", names=featurelist)

#设置显示格式
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
#df= df.replace('-',0)
print(df)
print('====')
df2 = df.service.value_counts()
print(df2)
df3 = df.proto.value_counts()
print(df3)
print('====')
#获取df.service,df.proto所有列名
lie = df.columns.values.tolist()
print(lie)
print('====')
#通过列名列表获取这一列
for i in range(len(lie)):
    print('第:',i,'列')
    print(df[lie[i]].value_counts())