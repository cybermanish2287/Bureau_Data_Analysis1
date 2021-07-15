#import os
#os.chdir(r"C:\Desktop\Learning\Courses\HFC_Data")

import pandas as pd
import numpy as np
from datetime import datetime

Target = pd.read_csv("Target.csv")#,nrows=2000)
Trade = pd.read_csv("Trades_Data.csv")#,nrows=2000)
Enquiry = pd.read_csv("Enquiry_Data.csv")#,nrows=2000)

Trade.dropna(subset = ['DateOpenedDisbursed'], inplace= True)
Trade.reset_index(inplace=True, drop = True)

Enquiry.dropna(subset=['EnquiryPurpose','DateofEnquiry'],inplace=True)
Enquiry.reset_index(drop=True, inplace=True)

def map_Account(x):
    if x in ([9,11,12,17,33,34,38,40,50,51,52,53,54,55,56,57,58,59,61]):
        return 'Business'
    elif x in ([2,42,44]):
        return 'Housing'
    elif x in ([3]):
        return 'LAP'
    elif x in ([7]):
        return 'GL'
    elif x in ([5,6,8,15,37,41]):
        return 'PL'
    elif x in ([1,13,32]):
        return 'Vehicle'
    elif x in ([10,31,35,36]):
        return 'Credit_Card'
    return 'Others'

Trade['Loan_Category'] = Trade['AccountType'].apply(map_Account)

def Loan_Type(x):
    if x in ([1,2,3,4,7,11,13,14,15,17,31,32,33,34,40,41,42,43,44,50,51,52,53,54,55,56,57,58,59]):
        return 'Secured'
    elif x in ([0,5,6,8,9,10,12,16,35,36,37,38,39,61]):
        return 'Unsecured'
    return 'Others'
    
Trade['Loan_Type'] = Trade['AccountType'].apply(Loan_Type)

#Date when Loan started in CIBIL
Trade['MkrDt'] = pd.to_datetime(Trade['MkrDt'].apply(lambda x: x.split(':')[0]))

#Last Repayment Date
Trade['PaymentHistoryStartDate'] = pd.to_datetime(Trade['PaymentHistoryStartDate'].apply(lambda x: str(x).zfill(8)), format = '%d%m%Y')

# Month over Books
Trade['Pull_Start_Month'] = 12*(Trade['MkrDt'].dt.year - Trade['PaymentHistoryStartDate'].dt.year) +(Trade['MkrDt'].dt.month - Trade['PaymentHistoryStartDate'].dt.month)


Trade['Helper'] = Trade['Pull_Start_Month'].apply(lambda x: 3*x*'0')
Trade['PaymentHistory1'].fillna('', inplace = True)
Trade['PaymentHistory2'].fillna('', inplace = True)
Trade['PH'] = Trade['PaymentHistory1'] + Trade['PaymentHistory2']
Trade['Resolved_PH'] = Trade['Helper'] + Trade['PH']


max_months = int(max(Trade['PH'].apply(len))/3)

for i in range(1,max_months+1):
    Trade['H' + str(i)] = Trade['Resolved_PH'].apply(lambda x: x[3*(i-1):3*(i)])
    Trade['H'+str(i)] = Trade['H'+str(i)].replace(to_replace = ['XXX','STD','SMA','SUB','LSS','DBT',''], value = [0,0,75,999,999,999,0]).astype(int)

pointer = Trade.columns.get_loc("H1")

for i in [3,6,9,12,18,24,36]:
    Trade['Max_DPD_'+str(i)+'M'] = Trade.iloc[:,pointer:pointer+i].max(axis=1)
    Trade['DPD_'+str(i)+'M_0'] = (Trade.iloc[:,pointer:pointer+i]>0).sum(axis=1)
    Trade['DPD_'+str(i)+'M_30'] = (Trade.iloc[:,pointer:pointer+i]>30).sum(axis=1)
    Trade['DPD_'+str(i)+'M_60'] = (Trade.iloc[:,pointer:pointer+i]>60).sum(axis=1)
    Trade['DPD_'+str(i)+'M_90'] = (Trade.iloc[:,pointer:pointer+i]>90).sum(axis=1)
    Trade['Fg_DPD_0_'+str(i)+'M'] = (Trade['DPD_'+str(i)+'M_0']>0).astype(int)
    Trade['Fg_DPD_30_'+str(i)+'M'] = (Trade['DPD_'+str(i)+'M_0']>30).astype(int)
    Trade['Fg_DPD_60_'+str(i)+'M'] = (Trade['DPD_'+str(i)+'M_0']>60).astype(int)
    Trade['Fg_DPD_90_'+str(i)+'M'] = (Trade['DPD_'+str(i)+'M_0']>90).astype(int)

cols = list(Trade.loc[:,'Max_DPD_3M':'Fg_DPD_90_36M'].columns)
Flag_fields = [item for item in cols if item.startswith('Fg')]
Max_fields = [item for item in cols if item.startswith('Max')]
Sum_fields = []
for item in cols:
    if item not in Flag_fields:
        if item not in Max_fields:
            Sum_fields.append(item)
            
Sum1 = Trade.pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=Sum_fields, aggfunc='sum')
Columns = pd.DataFrame(list(Sum1.columns))
Columns['col'] = Columns.iloc[:,0] +'_'+ Columns.iloc[:,1]
Sum1.columns = list(Columns['col'])


Sum2 = Trade.pivot_table(index=['Prospectno'], columns=['Loan_Type'], values=Sum_fields, aggfunc='sum')
Columns = pd.DataFrame(list(Sum2.columns))
Columns['col'] = Columns.iloc[:,0] +'_'+ Columns.iloc[:,1]
Sum2.columns = list(Columns['col'])


Max1 = Trade.pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=Max_fields, aggfunc='max')
Columns = pd.DataFrame(list(Max1.columns))
Columns['col'] = Columns.iloc[:,0] +'_'+ Columns.iloc[:,1]
Max1.columns = list(Columns['col'])

Max2 = Trade.pivot_table(index=['Prospectno'], columns=['Loan_Type'], values=Max_fields, aggfunc='max')
Columns = pd.DataFrame(list(Max2.columns))
Columns['col'] = Columns.iloc[:,0] +'_'+ Columns.iloc[:,1]
Max2.columns = list(Columns['col'])


for date in ['DateClosed', 'DateOpenedDisbursed']:
    Trade[date] = Trade[date].apply(lambda x: str(x).split('.')[0])
    Trade[date] = pd.to_datetime(Trade[date].apply(lambda x: str(x).zfill(8) if x !='nan' else x),format="%d%m%Y")


Trade['Month_Open'] = (Trade['MkrDt'].dt.month - Trade['DateOpenedDisbursed'].dt.month) + 12*(Trade['MkrDt'].dt.year - Trade['DateOpenedDisbursed'].dt.year)
Trade['Month_Close'] = (Trade['MkrDt'].dt.month - Trade['DateClosed'].dt.month) + 12*(Trade['MkrDt'].dt.year - Trade['DateClosed'].dt.year)

Open_Trade_3M = Trade[Trade['Month_Open']<=3].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Open_Trade_6M = Trade[Trade['Month_Open']<=6].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Open_Trade_9M = Trade[Trade['Month_Open']<=9].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Open_Trade_12M = Trade[Trade['Month_Open']<=12].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Open_Trade_18M = Trade[Trade['Month_Open']<=18].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Open_Trade_24M = Trade[Trade['Month_Open']<=24].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')


Open_Trade_3M.columns = ["Open"+ str(s2) + "_3M"  for (s1,s2) in Open_Trade_3M.columns]
Open_Trade_6M.columns = ["Open"+ str(s2) + "_6M"  for (s1,s2) in Open_Trade_6M.columns]
Open_Trade_9M.columns = ["Open"+ str(s2) + "_9M"  for (s1,s2) in Open_Trade_9M.columns]
Open_Trade_12M.columns = ["Open"+ str(s2) + "_12M"  for (s1,s2) in Open_Trade_12M.columns]
Open_Trade_18M.columns = ["Open"+ str(s2) + "_18M"  for (s1,s2) in Open_Trade_18M.columns]
Open_Trade_24M.columns = ["Open"+ str(s2) + "_24M"  for (s1,s2) in Open_Trade_24M.columns]    

Trade_Final = pd.concat([Open_Trade_3M,Open_Trade_6M,Open_Trade_9M,Open_Trade_12M,Open_Trade_18M,Open_Trade_24M, Max1,Max2,Sum1,Sum2], axis=1, sort=False).fillna(0)



Enquiry['Loan_Category'] = Enquiry['EnquiryPurpose'].apply(map_Account)

Enquiry['DateofEnquiry'] = pd.to_datetime(Enquiry['DateofEnquiry'].apply(lambda x: str(x).zfill(8)), format="%d%m%Y")

Enquiry['MkrDt']= pd.to_datetime(Enquiry['MkrDt'].apply(lambda x: x.split(":")[0]))
Enquiry['Month'] = (Enquiry['MkrDt'].dt.month - Enquiry['DateofEnquiry'].dt.month) + 12*(Enquiry['MkrDt'].dt.year - Enquiry['DateofEnquiry'].dt.year)


Enquiry_3M = Enquiry[Enquiry['Month']<=3].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Enquiry_6M = Enquiry[Enquiry['Month']<=6].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Enquiry_9M = Enquiry[Enquiry['Month']<=9].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Enquiry_12M = Enquiry[Enquiry['Month']<=12].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Enquiry_18M = Enquiry[Enquiry['Month']<=18].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')
Enquiry_24M = Enquiry[Enquiry['Month']<=24].pivot_table(index=['Prospectno'], columns=['Loan_Category'], values=['SrNo'], aggfunc='count')


Enquiry_3M.columns = ["Enq"+ str(s2) + "_3M"  for (s1,s2) in Enquiry_3M.columns]
Enquiry_6M.columns = ["Enq"+ str(s2) + "_6M"  for (s1,s2) in Enquiry_6M.columns]
Enquiry_9M.columns = ["Enq"+ str(s2) + "_9M"  for (s1,s2) in Enquiry_9M.columns]
Enquiry_12M.columns = ["Enq"+ str(s2) + "_12M"  for (s1,s2) in Enquiry_12M.columns]
Enquiry_18M.columns = ["Enq"+ str(s2) + "_18M"  for (s1,s2) in Enquiry_18M.columns]
Enquiry_24M.columns = ["Enq"+ str(s2) + "_24M"  for (s1,s2) in Enquiry_24M.columns]



Enquiry_Final = pd.concat([Enquiry_3M,Enquiry_6M,Enquiry_9M,Enquiry_12M,Enquiry_18M,Enquiry_24M], axis=1).fillna(0)

trade = list(Trade_Final.loc[:,'OpenBusiness_3M':'OpenVehicle_24M'].columns)
enq = list(Enquiry_Final.loc[:,'EnqBusiness_3M':'EnqVehicle_24M'].columns)

trade_enq = pd.DataFrame()
for i in range(len(trade)):
    trade_enq['Enq_Trd_'+trade[i][4:]] = (Enquiry_Final[enq[i]]/Trade_Final[trade[i]]).replace(to_replace = [np.nan,np.inf], value = [0,0])

Final_Applicant_Data = pd.concat([Trade_Final,Enquiry_Final,trade_enq], axis=1).fillna(0)
Final_Applicant_Data.to_csv('Final_Applicant_Data.csv')
