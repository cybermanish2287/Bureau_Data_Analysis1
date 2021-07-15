import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
#from tqdm import tqdm

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

target = pd.read_csv("Target.csv", nrows = 1000)
trade = pd.read_csv("Trades_Data.csv")
enquiry = pd.read_csv("Enquiry_Data.csv")

enquiry.dropna(subset=['EnquiryPurpose','DateofEnquiry'],inplace=True)
enquiry.reset_index(drop=True, inplace=True)


def Acct_category(x):
    if x in [9,11,12,17,33,34,38,40,50,51,52,53,54,55,56,57,58,59,61]: return "Business_Use"
    elif x in [2,42,44]: return "Housing"
    elif x in [3]: return "LAP"
    elif x in [7]: return "GL"
    elif x in [5,6,8,15,37,41]: return "PL"
    elif x in [1,13,32]: return "Vehicle"
    elif x in [10,31,35,36]: return "Credit_Card"
    else: return "Others"
       
def Account_Type(x):
    if x in [1,2,3,4,7,11,13,14,15,17,31,32,33,34,40,41,42,43,44,50,51,52,53,54,55,56,57,58,59]: return "Secured"
    else: return "Unsecured";
   
def Acct_cat_grp(x):
    if x in [2,3,42,44]: return "Cat_1"
    elif x in [1,5,17,32,33,34]: return "Cat_2"
    elif x in [4,7,9,11,12,13,14]: return "Cat_3"
    else: return "Cat_4"



enquiry['Acct_category'] = enquiry['EnquiryPurpose'].apply(Acct_category)
enquiry['Account_Type'] = enquiry['EnquiryPurpose'].apply(Account_Type)
enquiry['Acct_cat_grp'] = enquiry['EnquiryPurpose'].apply(Acct_cat_grp)

enquiry['CIBIL_Date'] = enquiry['MkrDt'].apply(lambda x:dt.strptime(str(x)[0:18],"%d%b%Y:%H:%M:%S"))

enquiry['EnqDate'] = enquiry['DateofEnquiry'].astype(int).apply(lambda x: dt.strptime(str(x).zfill(8),"%d%m%Y"))
enquiry['Days_diff'] = (enquiry['CIBIL_Date']- enquiry['EnqDate']).dt.days

enquiry['Fg']= ((enquiry['EnquiringMemberShortName'].isin(["IIHFL","IIFL"])) & (enquiry['EnquiryPurpose'].isin(["02","03","42","44"]))).astype(int)
enquiry = enquiry[enquiry['Fg'] != 1].drop('Fg',axis=1)

#enquiry.groupby(['Prospectno','ApplicantName'])['SrNo'].count().count()

enq_3M = enquiry[enquiry['Days_diff'] <= 90].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
enq_6M = enquiry[enquiry['Days_diff'] <= 180].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
enq_9M = enquiry[enquiry['Days_diff'] <= 270].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
enq_12M = enquiry[enquiry['Days_diff'] <= 365].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
enq_24M = enquiry[enquiry['Days_diff'] <= 730].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
enq_36M = enquiry[enquiry['Days_diff'] <= 1095].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
enq_LFT = enquiry.pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')

enq_3M.columns =["num_enq_"+str(s1)+"_3M" for s1 in enq_3M.columns]
enq_6M.columns =["num_enq_"+str(s1)+"_6M" for s1 in enq_6M.columns]
enq_9M.columns =["num_enq_"+str(s1)+"_9M" for s1 in enq_9M.columns]
enq_12M.columns =["num_enq_"+str(s1)+"_12M" for s1 in enq_12M.columns]
enq_24M.columns =["num_enq_"+str(s1)+"_24M" for s1 in enq_24M.columns]
enq_36M.columns =["num_enq_"+str(s1)+"_36M" for s1 in enq_36M.columns]
enq_LFT.columns =["num_enq_"+str(s1)+"_LFT" for s1 in enq_LFT.columns]

enq_account_type_3M = enquiry[enquiry['Days_diff'] <= 90].pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')
enq_account_type_6M = enquiry[enquiry['Days_diff'] <= 180].pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')
enq_account_type_9M = enquiry[enquiry['Days_diff'] <= 270].pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')
enq_account_type_12M = enquiry[enquiry['Days_diff'] <= 365].pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')
enq_account_type_24M = enquiry[enquiry['Days_diff'] <= 730].pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')
enq_account_type_36M = enquiry[enquiry['Days_diff'] <= 1095].pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')
enq_account_type_LFT = enquiry.pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values='SrNo', aggfunc='count')

enq_account_type_3M.columns =["num_enq_"+str(s1)+"_3M" for s1 in enq_account_type_3M.columns]
enq_account_type_6M.columns =["num_enq_"+str(s1)+"_6M" for s1 in enq_account_type_6M.columns]
enq_account_type_9M.columns =["num_enq_"+str(s1)+"_9M" for s1 in enq_account_type_9M.columns]
enq_account_type_12M.columns =["num_enq_"+str(s1)+"_12M" for s1 in enq_account_type_12M.columns]
enq_account_type_24M.columns =["num_enq_"+str(s1)+"_24M" for s1 in enq_account_type_24M.columns]
enq_account_type_36M.columns =["num_enq_"+str(s1)+"_36M" for s1 in enq_account_type_36M.columns]
enq_account_type_LFT.columns =["num_enq_"+str(s1)+"_LFT" for s1 in enq_account_type_LFT.columns]

Enq = pd.concat([enq_3M,enq_6M,enq_9M,enq_12M,enq_24M,enq_36M,enq_LFT,enq_account_type_3M, enq_account_type_6M, enq_account_type_9M, \
                 enq_account_type_12M, enq_account_type_24M, enq_account_type_36M, enq_account_type_LFT],axis=1).fillna(0)

# TRADE DATASET VARIABLE CREATION

trade.dropna(subset = ['DateOpenedDisbursed'], inplace=True)
trade.reset_index(drop=True, inplace=True)
trade['PaymentHistory2'].fillna('', inplace=True)
trade['PaymentHistory1'].fillna('', inplace=True)
trade['PH'] = trade['PaymentHistory1'] + trade['PaymentHistory2']

tot_months = int(max(trade.PH.apply(len))/3)

for i in range(1,tot_months+1):
    trade['H_'+str(i)] = trade['PH'].str[3*(i-1):3*i]
    trade['H_M_'+str(i)] = trade['H_'+str(i)].replace(to_replace = ['XXX','STD','','SMA','LSS','DBT','SUB'], value = [0,0,0,75,999,999,999]).astype(int)
    del trade['H_'+str(i)]

#from datetime import datetime as dt
trade['CIBIL_Date'] = trade['MkrDt'].apply(lambda x:dt.strptime(str(x)[0:18],"%d%b%Y:%H:%M:%S"))

trade['Last_Repayment_Date'] = trade['PaymentHistoryStartDate'].astype(str).apply(lambda x: dt.strptime(x.zfill(8),"%d%m%Y"))
trade['Repayment_month_diff'] = (trade['CIBIL_Date'].dt.month - trade['Last_Repayment_Date'].dt.month) + 12*(trade['CIBIL_Date'].dt.year - trade['Last_Repayment_Date'].dt.year)

trade['OpenDate'] = trade['DateOpenedDisbursed'].astype(int).apply(lambda x: dt.strptime(str(x).zfill(8),"%d%m%Y"))
trade['Month_diff'] = (trade['CIBIL_Date'].dt.month - trade['OpenDate'].dt.month) + 12*(trade['CIBIL_Date'].dt.year - trade['OpenDate'].dt.year)
#trade['Open_month_band'] = trade['Month_diff'].apply(lambda x: '<3M' if x<=3 else '4-6M' if x<=6 else '7-9M' if x<=9 else '10-12M' if x<=12 else '13-18M' if x<=18 else '19-24M' if x<=24 else '25-36M' if x<=36 else '>36M')

#def Acct_category(x):
#    if x in [9,11,12,17,33,34,38,40,50,51,52,53,54,55,56,57,58,59,61]: return "Business_Use"
#    elif x in [2,42,44]: return "Housing"
#    elif x in [3]: return "LAP"
#    elif x in [7]: return "GL"
#    elif x in [5,6,8,15,37,41]: return "PL"
#    elif x in [1,13,32]: return "Vehicle"
#    elif x in [10,31,35,36]: return "Credit_Card"
#    else: return "Others"
#       
#def Account_Type(x):
#    if x in [1,2,3,4,7,11,13,14,15,17,31,32,33,34,40,41,42,43,44,50,51,52,53,54,55,56,57,58,59]: return "Secured"
#    else: return "Unsecured";
#   
#def Acct_cat_grp(x):
#    if x in [2,3,42,44]: return "Cat_1"
#    elif x in [1,5,17,32,33,34]: return "Cat_2"
#    elif x in [4,7,9,11,12,13,14]: return "Cat_3"
#    else: return "Cat_4"

#pd.crosstab(trade['AccountType'], trade['Acct_category'])

trade['Acct_category'] = trade['AccountType'].apply(Acct_category)
trade['Account_Type'] = trade['AccountType'].apply(Account_Type)
trade['Acct_cat_grp'] = trade['AccountType'].apply(Acct_cat_grp)

trade['Fg_Default'] = trade["SuitFiledWilfulDefault"].fillna(0).astype(int).isin([1,2,3]) | trade["WrittenoffandSettledStatus"].fillna(0).astype(int).isin([2,3,4,6,8,9])




#trade['New_PH'] = trade['PH'].apply(lambda x: x.ljust(108, '0'))
trade['Add_PH'] = trade['Repayment_month_diff'].apply(lambda x: '0'*min(x*3,108))
trade['Resolved_PH'] = trade['Add_PH'] + trade['PH']

for i in range(1,37):
    trade['H_'+str(i)] = trade['Resolved_PH'].str[3*(i-1):3*i]
    trade['H_M_'+str(i)] = trade['H_'+str(i)].replace(to_replace = ['XXX','STD','','SMA','LSS','DBT','SUB'], value = [0,0,0,75,999,999,999]).astype(int)
    del trade['H_'+str(i)]

for i in [3,6,9,12,18,24,36]:
    trade['month_to_consider_'+str(i)+'M'] = trade['Repayment_month_diff'].apply(lambda x: max(i-x,0))
    trade['Max_'+str(i)+'M'] = trade.iloc[:,45:(45+i)].max(axis=1)
    trade['DPD_'+str(i)+'M_0'] = (trade.iloc[:,45:(45+i)]>0).sum(axis=1)
    trade['DPD_'+str(i)+'M_30'] = (trade.iloc[:,45:(45+i)]>31).sum(axis=1)
    trade['DPD_'+str(i)+'M_60'] = (trade.iloc[:,45:(45+i)]>60).sum(axis=1)
    trade['DPD_'+str(i)+'M_90'] = (trade.iloc[:,45:(45+i)]>90).sum(axis=1)
    trade['Fg_DPD_0_'+str(i)+'M'] = (trade['Max_'+str(i)+'M']>0).astype(int)
    trade['Fg_DPD_30_'+str(i)+'M'] = (trade['Max_'+str(i)+'M']>31).astype(int)
    trade['Fg_DPD_60_'+str(i)+'M'] = (trade['Max_'+str(i)+'M']>60).astype(int)
    trade['Fg_DPD_90_'+str(i)+'M'] = (trade['Max_'+str(i)+'M']>90).astype(int)

sum_col = [col for col in trade if col.startswith(('Fg_DPD_','DPD_','month_to_consider_','Fg_Default'))]
ind_sum = trade.groupby(['Prospectno','ApplicantName'])[sum_col].sum()
#ind_sum_reset = ind_sum.reset_index()

max_col = [col for col in trade if col.startswith('Max_')]
ind_max = trade.groupby(['Prospectno','ApplicantName'])[max_col].max().fillna(0)

#check = ind_sum.join(ind_max)

Del_col = [col for col in trade if col.startswith('Fg_DPD_')]
Del_Acct_category = trade.pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values=Del_col, aggfunc=np.sum)
Del_Account_Type = trade.pivot_table(index=['Prospectno','ApplicantName'], columns='Account_Type', values=Del_col, aggfunc=np.sum)
Del_Acct_cat_grp = trade.pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values=Del_col, aggfunc=np.sum)

Del_Acct_category.columns =[str(s2)+"_"+str(s1) for (s1,s2) in Del_Acct_category.columns.tolist()]
Del_Account_Type.columns = [str(s2)+"_"+str(s1) for (s1,s2) in Del_Account_Type.columns.tolist()]
Del_Acct_cat_grp.columns = [str(s2)+"_"+str(s1) for (s1,s2) in Del_Acct_cat_grp.columns.tolist()]

Del_Var = pd.concat([Del_Acct_category,Del_Account_Type,Del_Acct_cat_grp,ind_sum,ind_max], axis=1).fillna(0)

Open_trade_3M = trade[trade['Month_diff'] <= 3].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_6M = trade[trade['Month_diff'] <= 6].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_9M = trade[trade['Month_diff'] <= 9].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_12M = trade[trade['Month_diff'] <= 12].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_18M = trade[trade['Month_diff'] <= 18].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_24M = trade[trade['Month_diff'] <= 24].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_36M = trade[trade['Month_diff'] <= 36].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')
Open_trade_LFT = trade.pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_category', values='SrNo', aggfunc='count')

Open_trade_3M.columns =["num_trade_"+str(s1)+"_3M" for s1 in Open_trade_3M.columns]
Open_trade_6M.columns =["num_trade_"+str(s1)+"_6M" for s1 in Open_trade_6M.columns]
Open_trade_9M.columns =["num_trade_"+str(s1)+"_9M" for s1 in Open_trade_9M.columns]
Open_trade_12M.columns =["num_trade_"+str(s1)+"_12M" for s1 in Open_trade_12M.columns]
Open_trade_18M.columns =["num_trade_"+str(s1)+"_18M" for s1 in Open_trade_18M.columns]
Open_trade_24M.columns =["num_trade_"+str(s1)+"_24M" for s1 in Open_trade_24M.columns]
Open_trade_36M.columns =["num_trade_"+str(s1)+"_36M" for s1 in Open_trade_36M.columns]
Open_trade_LFT.columns =["num_trade_"+str(s1)+"_LFT" for s1 in Open_trade_LFT.columns]

Open_trade_Cat_3M = trade[trade['Month_diff'] <= 3].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_6M = trade[trade['Month_diff'] <= 6].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_9M = trade[trade['Month_diff'] <= 9].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_12M = trade[trade['Month_diff'] <= 12].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_18M = trade[trade['Month_diff'] <= 18].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_24M = trade[trade['Month_diff'] <= 24].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_36M = trade[trade['Month_diff'] <= 36].pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')
Open_trade_Cat_LFT = trade.pivot_table(index=['Prospectno','ApplicantName'], columns='Acct_cat_grp', values='SrNo', aggfunc='count')

Open_trade_Cat_3M.columns =["num_trade_"+str(s1)+"_3M" for s1 in Open_trade_Cat_3M.columns]
Open_trade_Cat_6M.columns =["num_trade_"+str(s1)+"_6M" for s1 in Open_trade_Cat_6M.columns]
Open_trade_Cat_9M.columns =["num_trade_"+str(s1)+"_9M" for s1 in Open_trade_Cat_9M.columns]
Open_trade_Cat_12M.columns =["num_trade_"+str(s1)+"_12M" for s1 in Open_trade_Cat_12M.columns]
Open_trade_Cat_18M.columns =["num_trade_"+str(s1)+"_18M" for s1 in Open_trade_Cat_18M.columns]
Open_trade_Cat_24M.columns =["num_trade_"+str(s1)+"_24M" for s1 in Open_trade_Cat_24M.columns]
Open_trade_Cat_36M.columns =["num_trade_"+str(s1)+"_36M" for s1 in Open_trade_Cat_36M.columns]
Open_trade_Cat_LFT.columns =["num_trade_"+str(s1)+"_LFT" for s1 in Open_trade_Cat_LFT.columns]


Open_Trade = pd.concat([Open_trade_3M,Open_trade_6M,Open_trade_9M,Open_trade_12M,Open_trade_18M,Open_trade_24M,Open_trade_36M,Open_trade_LFT, \
                        Open_trade_Cat_3M,Open_trade_Cat_6M,Open_trade_Cat_9M,Open_trade_Cat_12M,Open_trade_Cat_18M,Open_trade_Cat_24M, \
                        Open_trade_Cat_36M,Open_trade_Cat_LFT],axis=1).fillna(0)

# Master Data as Applicant Level
App_Master = pd.concat([Open_Trade,Del_Var,Enq], axis=1).fillna(0)
Prospect_Master = App_Master.groupby('Prospectno').max()
Prospect_Master.to_csv('Prospect_Master.csv')