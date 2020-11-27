import pandas as pd
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

tran=transition_parameter(data)
emi=emission_parameter_UNK(data).rename(columns={'state':"Yj"})
parameter=pd.merge(tran,emi)
parameter=parameter.drop(['Count_Yi','Count_Yi_j','CountY','CountY+k','CountY_X'],axis=1)
parameter=parameter[['word','Yi','Yj','transition','emission']]
parameter['probability']=parameter['transition']*parameter['emission']
parameter=parameter.drop(['transition','emission'],axis=1) #Yi: current state, Yj: next state, probability:transition*emission 

dev_in=pd.read_csv('dev.in',sep=' ',names=['word'],skip_blank_lines=False) 
dev_in=dev_in.fillna('Nil')
dic={'word':['Nil']}
first_line=pd.DataFrame(dic)
dev_in=pd.concat([first_line,dev_in],ignore_index=True)

def viterbi(parameters,input_data):
    input_data=input_data['word'].to_list()
    for i in range(0,len(input_data)):
        if input_data[i] not in data['word'].to_list():
            input_data[i]="#UNK#"
    route=[]
    score=[]
    for j in range(0,len(state)):
        route.append([])
        route[-1].append('Nil')
        score.append(1)
        route[-1].append(state[j])
        data1=parameters.loc[parameters['Yi']=='Nil']
        for index,row in data1.iterrows():
            if row['Yj']==state[j]:
                score[j]=score[j]*row['probability']
    for k in range(2,len(input_data)):
        data1=parameters.loc[parameters['word']==input_data[k]]
        for l in range(0,len(route)):
            data2=data1.loc[parameters['Yi']==route[l][-1]]
            prob=data2['probability'].to_list()
            st=data2['Yj']
            p=max(prob)
            score[l]=score[l]*p
            index=prob.index(max(prob))
            route[l].append(st[index])
    index=score.index(max(score))
    result=route[index]
    return result

print(viterbi(parameter,dev_in))


    