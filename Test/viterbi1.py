import pandas as pd
import numpy as np

def viterbi(obs,states,trans_p,emit_p):
    v=[]
    fst=obs[0]
    vn=emit_p[fst]
    v.append(vn)
 
    for t in range(1,len(obs)):
        tp=emit_p[obs[t]]
        cc=[]
        for y in states:
            gv=v[t-1]*trans_p[y]*tp.loc[y]
            cc.append(gv.max())
        cc1=pd.Series(cc,index=states)
        v.append(cc1)
 
    result=[{'Nil':'Nil'}]
    for i in range(len(v)-2,-1,-1):
        b=list(result[-1])
        a=v[i]*trans_p[b[0]]
        p1=a.sort_values(ascending=False)
        p2=p1[:1]
        result.append(dict(p2))
    return result

data=pd.read_csv('train_EN',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
observation=data['word']
state=data['state']
states=list(set(state))

dev_in=pd.read_csv('dev.in',sep=' ',names=['word'],skip_blank_lines=False)
dic={'word':['Nil']}
first_line=pd.DataFrame(dic)
dev_in=pd.concat([first_line,dev_in],ignore_index=True)
dev_in_list=np.split(dev_in,dev_in[dev_in.isnull().all(1)].index)
for i in range(1,len(dev_in_list)):
    dev_in_list[i]=dev_in_list[i].fillna('Nil')
for j in range(0,len(dev_in_list)-1):
    dev_in_list[j]=pd.concat([dev_in_list[j],first_line],ignore_index=True)

for i in range(0,len(dev_in_list)):
    dev_in_list[i]=dev_in_list[i]['word'].to_list()
    for j in range(0,len(dev_in_list[i])):
        if dev_in_list[i][j] not in observation.to_list():
            dev_in_list[i][j]='#UNK#'
trans_p=pd.read_csv('transition parameter.csv',index_col=0)
emi_p=pd.read_csv('emission parameter.csv',index_col=0)

key=[]
for i in range(0,len(dev_in_list)):
    result=viterbi(dev_in_list[i],states,trans_p,emi_p)
    temp=[]
    for j in range(0,len(result)):
        k=list(result[j].keys())
        temp.append(k[0])
    temp=list(reversed(temp))
    key.append(temp)

with open('dev.p5.out','w') as f:
   for i in range(0,len(key)):
       for j in range(0,len(key[i])):
           f.write(key[i][j]+'\n')
f.close 