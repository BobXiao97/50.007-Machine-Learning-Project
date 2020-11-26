import pandas as pd
data=pd.read_csv('train',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
dic={'word':['Nil'],'state':['Nil']}
first_line=pd.DataFrame(dic)
data=pd.concat([first_line,data],ignore_index=True)

def transition_parameter(data):
    state=data['state']
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


