#将u4随机取10000（Analysis，Backdoor ，ShellcodeWorms全取，其余取1000-670-666-371-43），分为5000,5000分别做验证和测试。
from columns import columns
import pandas as pd
dffeature = pd.read_csv("feature.csv")
featurelist = dffeature['Name']
df = pd.read_csv("../../UNSW_datas/UNSW-NB15_4.csv", names=featurelist, low_memory=False)

#设置显示格式
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

#[1750 rows x 49 columns],10000-1750=8250
df1 = df.loc[df['attack_cat'].isin(['Analysis','Backdoor','Worms',' Shellcode '])]
#[438294 rows x 49 columns]
df2 = df.loc[~df['attack_cat'].isin(['Analysis','Backdoor','Worms',' Shellcode '])]
print('将攻击类型补全')
df2.loc[df2['Label']==0, 'attack_cat'] = 'Normal'

#[8250 rows x 49 columns]
df2_part = df2.sample(n=8250,  replace=False, weights=None, random_state=None, axis=0)

#[10000 rows x 49 columns]
df_valtestdata=pd.concat([df2_part,df1],axis=0,ignore_index=False)
print((df_valtestdata))
lie = df_valtestdata.columns.values.tolist()
print(len(lie))
print(lie)
for i in range(len(lie)):
    print('第:',i,'列  ',lie[i])
    print(df_valtestdata[lie[i]].value_counts())

df_valtestdata.to_csv('../UNSW_beforeclean/UNSW-4part-valtestdata.csv',index=False,header=False,encoding="utf_8_sig")
print('拆分UNSW-4part-valtestdata·============================================================')