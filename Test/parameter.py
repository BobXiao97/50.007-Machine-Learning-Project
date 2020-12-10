import pandas as pd
import copy
import numpy as np
data=pd.read_csv('train_EN',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
dic={'word':['Nil'],'state':['Nil']}
first_line=pd.DataFrame(dic)
data=pd.concat([first_line,data],ignore_index=True)
state=data['state']

yi=[]   #yi
yj=[]   #y(i+1)

for i in range(0,len(state)-2):
    if state[i+1]=='Nil':
        yi.append(state[i+1])
    else:
        yi.append(state[i]+' '+state[i+1])
    yj.append(state[i+2])
x={'Yi':yi,'Yj':yj}
data1=pd.DataFrame(x)
data2=data1.groupby('Yi').size().reset_index(name='Count_Yi')
data3=data1.groupby(['Yi','Yj']).size().reset_index(name='Count_Yi_j')
data4=pd.merge(data2,data3,on='Yi')
transition=data4[['Yi','Yj','Count_Yi','Count_Yi_j']]
transition['transition']=transition['Count_Yi_j']/transition['Count_Yi']
tran=transition.drop(['Count_Yi','Count_Yi_j'],axis=1)

yi=tran['Yi']
yj=tran['Yj']
yi=list(set(yi))
yj=list(set(yj))
a=len(yi)
b=len(yj)
zeros_tran=np.zeros((a,b))
for i in range(0,a):
    tran1=tran.loc[tran['Yi']==yi[i]]
    for j in range(0,b):
        tran2=tran1.loc[tran1['Yj']==yj[j]]
        tran_para=tran2['transition']
        if len(tran_para)==1:
            zeros_tran[i][j]=tran_para
tran_table=pd.DataFrame(zeros_tran,index=yi,columns=yj).sort_index(axis=0)
tran_table.to_csv('transition parameter 2nd.csv')

