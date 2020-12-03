import pandas as pd
import copy
import numpy as np
data=pd.read_csv('train',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
dic={'word':['Nil'],'state':['Nil']}
first_line=pd.DataFrame(dic)
data=pd.concat([first_line,data],ignore_index=True)
state=data['state']
def transition_parameter(data):
    yi=[]   #yi
    yj=[]   #y(i+1)
    for i in range(0,len(state)-1):
        yi.append(state[i])
        yj.append(state[i+1])
    x={'Yi':yi,'Yj':yj}
    data1=pd.DataFrame(x)
    data2=data1.groupby('Yi').size().reset_index(name='Count_Yi')
    data3=data1.groupby(['Yi','Yj']).size().reset_index(name='Count_Yi_j')
    data4=pd.merge(data2,data3,on='Yi')
    transition=data4[['Yi','Yj','Count_Yi','Count_Yi_j']]
    transition['transition']=transition['Count_Yi_j']/transition['Count_Yi']
    return transition

tran=transition_parameter(data)
tran=tran.drop(['Count_Yi','Count_Yi_j'],axis=1)
yi=tran['Yi']
yi=list(set(yi))
a=len(yi)
zeros_tran=np.zeros((a,a))
for i in range(0,a):
    tran1=tran.loc[tran['Yi']==yi[i]]
    for j in range(0,a):
        tran2=tran1.loc[tran1['Yj']==yi[j]]
        tran_para=tran2['transition']
        if len(tran_para)==1:
            zeros_tran[i][j]=tran_para
tran_table=pd.DataFrame(zeros_tran,index=yi,columns=yi).sort_index(axis=0)
tran_table.to_csv('transition parameter.csv')

states=list(set(state))
states.remove('Nil')
UNK_list=[]
for i in range(0,len(states)):
    UNK_list.append('#UNK#')
dic={'word':UNK_list,'state':states}
data_UNK=pd.DataFrame(dic)

def emission_parameter_UNK(data):
    data=pd.concat([data,data_UNK])
    data1=data['state'].value_counts().reset_index(name='CountY')
    data1=data1.rename(columns={'index':'state'})
    data2=data.groupby(['word','state']).size().reset_index(name='CountY_X')
    data=pd.merge(data1,data2)
    data=data[['word','state','CountY','CountY_X']]
    data['CountY+k']=data['CountY']-0.5
    data['emission']=data['CountY_X']/data['CountY+k']
    for i in range(0,len(data['word'])):
        if data['word'][i]=='#UNK#':
            data['emission'][i]=data['emission'][i]/2
    return data

emi=emission_parameter_UNK(data)
emi=emi.drop(['CountY','CountY_X','CountY+k'],axis=1)
word=emi['word']
word=list(set(word))
b=len(word)
zeros_emi=np.zeros((a,b))
for i in range(0,a):
    emi1=emi.loc[emi['state']==yi[i]]
    for j in range(0,b):
        emi2=emi1.loc[emi1['word']==word[j]]
        emi_para=emi2['emission']
        if len(emi_para)==1:
            zeros_emi[i][j]=emi_para
emi_table=pd.DataFrame(zeros_emi,index=yi,columns=word).sort_index(axis=0)
emi_table.to_csv('emission parameter.csv')


