#将UNSW-4part-valtestdata.csv拆分为UNSW-4part-val.csv,UNSW-4part-test.csv,存入UNSW-beforeclean

from columns import columns
import pandas as pd
dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df = pd.read_csv("../UNSW_beforeclean/UNSW-4part-valtestdata.csv", names=featurelist, low_memory=False)


#设置显示格式
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

#打乱行的顺序
df = df.sample(frac=1).reset_index(drop=True)

df1 = df[:5000][:]
lie = df1.columns.values.tolist()
print(len(lie))
print(lie)
for i in range(45,len(lie)):
    print('第:',i,'列  ',lie[i])
    print(df1[lie[i]].value_counts())
df2 = df[5000:][:]
lie = df2.columns.values.tolist()
print(len(lie))
print(lie)
for i in range(45,len(lie)):
    print('第:',i,'列  ',lie[i])
    print(df2[lie[i]].value_counts())

df1.to_csv('../UNSW_beforeclean/UNSW-4part-valdata.csv',index=False,header=False,encoding="utf_8_sig")
df2.to_csv('../UNSW_beforeclean/UNSW-4part-testdata.csv',index=False,header=False,encoding="utf_8_sig")
print('split fished!')
print('清洗=====================================================================')