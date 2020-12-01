import pandas as pd
import numpy as np
def evaluation(pred,answer):
    x=pred['state'].to_list()
    y=answer['state'].to_list()
    correct=0
    for i in range(0,len(y)):
        if x[i]==y[i]:
            if x[i][0]!='O':
                correct+=1    

    total_pred=0
    for j in range(0,len(x)):
        if x[j][0]!='O':
            total_pred+=1
            
    total_answer=0
    for k in range(0,len(y)):
        if y[k][0]!='O':
            total_answer+=1
    precision=correct/total_pred
    recall=correct/total_answer
    F=2/((1/precision)+(1/recall))
    return precision,recall,F

pred=pd.read_csv('dev.p3.out',sep=' ',names=['state'])
ans=pd.read_csv('dev.out',sep=' ',names=['word','state'],skip_blank_lines=False)
dic={'word':['Nil'],'state':['Nil']}
first_line=pd.DataFrame(dic)
ans=pd.concat([first_line,ans],ignore_index=True)
ans_list=np.split(ans,ans[ans.isnull().all(1)].index)
for i in range(1,len(ans_list)):
    ans_list[i]=ans_list[i].fillna('Nil')
for j in range(0,len(ans_list)-1):
    ans_list[j]=pd.concat([ans_list[j],first_line],ignore_index=True)
answer=[]
for k in range(0,len(ans_list)):
    for l in range(0,len(ans_list[k]['state'].to_list())):
        answer.append(ans_list[k]['state'].to_list()[l])
answer=pd.DataFrame(answer,columns=['state'])
precision,recall,F=evaluation(pred,answer)
print('Precision:'+str(precision))
print('Recall:'+str(recall))
print('F:'+str(F))  


