import pandas as pd
#获取特征名列表作为列名
from columns import columns
import numpy as np
listname = ['sport', 'dsport', 'proto_3pc', 'proto_a/n', 'proto_aes-sp3-d', 'proto_any', 'proto_argus', 'proto_aris', 'proto_arp', 'proto_ax.25', 'proto_bbn-rcc', 'proto_bna', 'proto_br-sat-mon', 'proto_cbt', 'proto_cftp', 'proto_chaos', 'proto_compaq-peer', 'proto_cphb', 'proto_cpnx', 'proto_crtp', 'proto_crudp', 'proto_dcn', 'proto_ddp', 'proto_ddx', 'proto_dgp', 'proto_egp', 'proto_eigrp', 'proto_emcon', 'proto_encap', 'proto_esp', 'proto_etherip', 'proto_fc', 'proto_fire', 'proto_ggp', 'proto_gmtp', 'proto_gre', 'proto_hmp', 'proto_i-nlsp', 'proto_iatp', 'proto_ib', 'proto_icmp', 'proto_idpr', 'proto_idpr-cmtp', 'proto_idrp', 'proto_ifmp', 'proto_igmp', 'proto_igp', 'proto_il', 'proto_ip', 'proto_ipcomp', 'proto_ipcv', 'proto_ipip', 'proto_iplt', 'proto_ipnip', 'proto_ippc', 'proto_ipv6', 'proto_ipv6-frag', 'proto_ipv6-no', 'proto_ipv6-opts', 'proto_ipv6-route', 'proto_ipx-n-ip', 'proto_irtp', 'proto_isis', 'proto_iso-ip', 'proto_iso-tp4', 'proto_kryptolan', 'proto_l2tp', 'proto_larp', 'proto_leaf-1', 'proto_leaf-2', 'proto_merit-inp', 'proto_mfe-nsp', 'proto_mhrp', 'proto_micp', 'proto_mobile', 'proto_mtp', 'proto_mux', 'proto_narp', 'proto_netblt', 'proto_nsfnet-igp', 'proto_nvp', 'proto_ospf', 'proto_pgm', 'proto_pim', 'proto_pipe', 'proto_pnni', 'proto_pri-enc', 'proto_prm', 'proto_ptp', 'proto_pup', 'proto_pvp', 'proto_qnx', 'proto_rdp', 'proto_rsvp', 'proto_rtp', 'proto_rvd', 'proto_sat-expak', 'proto_sat-mon', 'proto_sccopmce', 'proto_scps', 'proto_sctp', 'proto_sdrp', 'proto_secure-vmtp', 'proto_sep', 'proto_skip', 'proto_sm', 'proto_smp', 'proto_snp', 'proto_sprite-rpc', 'proto_sps', 'proto_srp', 'proto_st2', 'proto_stp', 'proto_sun-nd', 'proto_swipe', 'proto_tcf', 'proto_tcp', 'proto_tlsp', 'proto_tp++', 'proto_trunk-1', 'proto_trunk-2', 'proto_ttp', 'proto_udp', 'proto_udt', 'proto_unas', 'proto_uti', 'proto_vines', 'proto_visa', 'proto_vmtp', 'proto_vrrp', 'proto_wb-expak', 'proto_wb-mon', 'proto_wsn', 'proto_xnet', 'proto_xns-idp', 'proto_xtp', 'proto_zero', 'state_ACC', 'state_CLO', 'state_CON', 'state_ECO', 'state_ECR', 'state_FIN', 'state_INT', 'state_MAS', 'state_PAR', 'state_REQ', 'state_RST', 'state_TST', 'state_TXD', 'state_URH', 'state_URN', 'state_no', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service_defservice', 'service_dhcp', 'service_dns', 'service_ftp', 'service_ftp-data', 'service_http', 'service_irc', 'service_pop3', 'service_radius', 'service_smtp', 'service_snmp', 'service_ssh', 'service_ssl', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'attack_cat']
print(len(listname))
dfw = pd.DataFrame(listname, columns=['Name'])
featurelist = dfw['Name']
print(featurelist)
df = pd.read_csv("UNSW-NB15_1234_after-onehot.csv", names=featurelist, low_memory=False)

#设置显示格式
pd.set_option('display.max_rows', 230)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

lie = df.columns.values.tolist()
print(len(lie))
print(lie)
print(df)

print('标准化=======================================================================')
label_df = df[['attack_cat']]  #获取标签
df = df.drop(labels=['attack_cat'],axis=1) #删除最后一列
# df['service_dfservice']=df['service_dfservice'].astype(float)
# df = df[:][:].astype(float)
lietotal = df.columns.values.tolist()   #不含attacka_cat
#标准化
print(df)
# lie = df.columns.values.tolist()
# for i in range(len(lie)):
#     print('第:',i,'列  ',lie[i])
#     print(df[lie[i]].value_counts())
# df.info()
df.dtypes
print(df.dtypes)
df[['dsport']] = df[['dsport']].astype(float)
print(df.dtypes)
df['dsport'] = df['dsport'].apply(pd.to_numeric,errors ='coerce')
print(df.dtypes)
from  sklearn import preprocessing
scaler=preprocessing.StandardScaler()
df=scaler.fit_transform(np.array(df))                  #已经变成了一个array数组
df = pd.DataFrame(df,columns=lietotal)              #传入list作为列名，featurelist是一列dataframe
print(df)

#拼接
df=pd.concat([df,label_df],axis=1,ignore_index=False)
print(len(label_df))