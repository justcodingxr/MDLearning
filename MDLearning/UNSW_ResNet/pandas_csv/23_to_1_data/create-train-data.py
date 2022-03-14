#将2,3，的后面四类数据加入4构建训练集a，将a中大量的normal数据删除一些，将a分为训练集和验证集，1选取部分作为测试集。实现样本均匀
from columns import columns
import pandas as pd
dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df = pd.read_csv("../../UNSW_datas/UNSW-NB15_2.csv", names=featurelist, low_memory=False)


#设置显示格式
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

print('数据合并得到新的训练集')

#UNSW-NB15_2.csv,选取多行'Analysis','Backdoor','Worms',' Shellcode '[1342 rows x 49 columns]
df2 = df.loc[df['attack_cat'].isin(['Analysis','Backdoor','Worms',' Shellcode '])]
# print(df2)



#UNSW-NB15_3.csv,选取多行'Analysis','Backdoor','Worms',' Shellcode '[2292 rows x 49 columns]
df = pd.read_csv("../../UNSW_datas/UNSW-NB15_3.csv", names=featurelist, low_memory=False)
df3 = df.loc[df['attack_cat'].isin(['Analysis','Backdoor','Worms',' Shellcode '])]
# print(df3)

#将df2,df3合并,[3634 rows x 49 columns]
df23=pd.concat([df2,df3],axis=0,ignore_index=False)
lie = df23.columns.values.tolist()
print(lie)
print(len(lie))


#先将UNSW-NB15_1.csv的Normal数据随机抽取n个暂存df4，再将Normal全部删除，
# 将df4_part，和UNSW-NB15_1，df（UNSW-NB15_2.csv,UNSW-NB15_3.csv选取多行）合并

df_UNSW1 = pd.read_csv("../../UNSW_datas/UNSW-NB15_1.csv", names=featurelist, low_memory=False)  #[700001 rows x 49 columns]
#选择label为0的正常数据[677786 rows x 49 columns]
df4 = df_UNSW1.loc[df_UNSW1['Label']==0]

#抽取df4,[10000 rows x 49 columns]
df4_part =df4.sample(n=10000,  replace=False, weights=None, random_state=None, axis=0)

#[22215 rows x 49 columns]
df_UNSW1 = df_UNSW1.loc[df_UNSW1['Label'].isin([1])]

#[35849 rows x 49 columns]
df_traindata=pd.concat([df4_part,df_UNSW1,df23],axis=0,ignore_index=False)
#获取df.service,df.proto所有列名
lie = df_traindata.columns.values.tolist()
print(len(lie))
print(lie)
print(df_traindata)
df_traindata.to_csv('../UNSW_beforeclean/UNSW-23to1-train-raw.csv',index=False,header=False,encoding="utf_8_sig")
print('清洗·============================================================')
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