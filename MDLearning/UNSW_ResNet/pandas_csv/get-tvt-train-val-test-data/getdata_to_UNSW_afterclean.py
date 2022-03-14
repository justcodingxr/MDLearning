import pandas as pd
#获取特征名列表作为列名
from columns import columns


df1 = pd.read_csv("../../UNSW_afterclean/UNSW-NB15_1234_simple_train_after-onehot-standard.csv", low_memory=False)
df2 = pd.read_csv("../../UNSW_afterclean/UNSW-NB15_1234_simple_val_after-onehot-standard.csv", low_memory=False)
df3 = pd.read_csv("../../UNSW_afterclean/UNSW-NB15_1234_simple_test_after-onehot-standard.csv", low_memory=False)

#设置显示格式
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

print(df1.dtypes)
print(df2.dtypes)
print(df3.dtypes)

df1.to_csv('../../UNSW_afterclean/train_raw.csv',index=False,header=False)
df2.to_csv('../../UNSW_afterclean/val_raw.csv',index=False,header=False)
df3.to_csv('../../UNSW_afterclean/test_raw.csv',index=False,header=False)
